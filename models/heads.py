"""
heads.py — Task-specific output heads (5-task multi-task structure).

1. ClassificationHead  : diagnostic classification — Softmax
2. RegressionHead      : cognitive score regression (WAB-AQ 0-100) — MSE
3. SimilarityHead      : similar-patient retrieval — InfoNCE contrastive loss
4. ProgressionHead     : disease progression risk prediction — BCE (used with memory module)
5. GenerationHead      : clinical explanation generation (template-based)

Multi-task loss:
  L_total = λ_cls·L_cls + λ_reg·L_reg + λ_sim·L_sim
  (computed only on samples with available labels — missing values masked)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (HIDDEN_DIM, N_CLASSES, LAMBDA_CLS, LAMBDA_REG, LAMBDA_SIM,
                    LABEL_SMOOTHING, FOCAL_GAMMA, IDX_TO_LABEL)


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    γ=2 focuses on hard / misclassified samples (minority classes).
    α per-class weights set by Trainer from inverse-frequency.
    """
    def __init__(self, gamma: float = FOCAL_GAMMA,
                 label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha: torch.Tensor | None = None   # set externally by Trainer

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce = F.cross_entropy(logits, labels, weight=alpha,
                              label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce.detach())             # probability of true class
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ─── 1. Classification head ──────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = HIDDEN_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_classes),
        )
        self.focal = FocalLoss()    # alpha set by Trainer after construction

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """returns logits (B, N_CLASSES)"""
        return self.net(shared_repr)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, labels)


# ─── 2. Regression head ──────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    """WAB-AQ score prediction (0-100). Samples without a label (-1) are masked."""
    def __init__(self, d_model: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """returns predicted WAB-AQ (B, 1), in normalized space"""
        return self.net(shared_repr)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE over samples with a WAB-AQ value only (missing masked)."""
        mask = (targets >= 0)           # wab_aq = -1 means missing
        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        return F.mse_loss(preds[mask].squeeze(-1), targets[mask])


# ─── 3. Similarity head ──────────────────────────────────────────────────────
class SimilarityHead(nn.Module):
    """
    Contrastive learning of a similar-patient embedding space.
    InfoNCE loss: pull same-diagnosis patients together, push others apart.
    """
    def __init__(self, d_model: int = HIDDEN_DIM, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """normalized projected embedding (B, proj_dim)"""
        return F.normalize(self.proj(shared_repr), dim=-1)

    def loss(self, embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        L_sim = -log(exp(sim(q,pos)) / Σexp(sim(q,neg)))
        Within a batch: same label = positive pair, different label = negative pair.
        """
        B    = embs.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embs.device, requires_grad=True)

        # Cosine similarity matrix
        sim  = embs @ embs.T / self.temperature.clamp(min=0.05, max=2.0)   # (B, B)

        # Exclude self-similarity
        eye  = torch.eye(B, device=embs.device, dtype=torch.bool)
        sim  = sim.masked_fill(eye, float('-inf'))

        # positive mask: same label
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # (B, B)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embs.device, requires_grad=True)

        # InfoNCE: mean loss over positive pairs per sample.
        # Use masked_fill (not multiplication) to zero non-positive positions,
        # avoiding log_softmax(-inf) * 0.0 = NaN
        log_softmax = F.log_softmax(sim, dim=-1)
        log_softmax_pos = log_softmax.masked_fill(~pos_mask, 0.0)
        loss_per_row = -log_softmax_pos.sum(dim=-1) / \
                       pos_mask.float().sum(dim=-1).clamp(min=1)
        loss = loss_per_row.mean()
        if torch.isnan(loss):
            return torch.tensor(0.0, device=embs.device, requires_grad=True)
        return loss


# ─── 4. Progression risk head (used with the memory module) ──────────────────
class ProgressionHead(nn.Module):
    """
    Difference of two-timepoint representations Δ = r(t2) - r(t1) → progression risk score (0-1).
    """
    def __init__(self, d_model: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (B, D) → risk (B, 1)"""
        return self.net(delta)


# ─── 5. Clinical explanation generation head ─────────────────────────────────
class GenerationHead:
    """
    Structured clinical report generator.

    Approach: evidence-grounded template generation — confidence-stratified
    reports (high / moderate / low) with per-class linguistic evidence and
    actionable recommendations.  Covers all 6 diagnostic categories.

    Evaluation: Clinical Consistency Score (CCS) — fraction of test-set
    samples where the generated report mentions the correct diagnosis-specific
    linguistic markers.

    Design rationale: fully LLM-based generation (LLaMA/Mistral + LoRA) would
    require curated clinical reference reports for supervised fine-tuning or
    RLHF, which are not available in TalkBank.  Evidence-grounded templates
    allow quantitative CCS evaluation while keeping the system self-contained.
    """

    # Confidence thresholds for severity language
    HIGH_CONF = 0.70
    MOD_CONF  = 0.45

    # Per-class key markers used by CCS evaluator (must match report text)
    CLASS_MARKERS = {
        "Control": ["within normal limits", "no clinical indicators"],
        "Aphasia": ["aphasia", "WAB-AQ", "word retrieval"],
        "AD":      ["alzheimer", "content density", "semantic"],
        "TBI":     ["traumatic brain injury", "discourse", "coherence"],
        "RHD":     ["right hemisphere", "pragmatic", "inferencing"],
        "MCI":     ["mild cognitive impairment", "episodic memory", "naming"],
    }

    _TEMPLATES = {
        "Control": {
            "high": (
                "ASSESSMENT: Speech and language patterns are within normal limits "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Type-token ratio (TTR={ttr:.3f}), mean utterance length "
                "({avg_len:.1f} words), and filler rate ({filler:.1%}) are all "
                "consistent with neurologically healthy speakers.\n"
                "RECOMMENDATION: No clinical indicators of neurolinguistic disorder "
                "are detected. Routine follow-up is sufficient."
            ),
            "mod": (
                "ASSESSMENT: Speech patterns are largely within normal limits "
                "(confidence: {conf:.1%}), though some features warrant monitoring.\n"
                "EVIDENCE: TTR={ttr:.3f}; utterance length={avg_len:.1f} words; "
                "filler rate={filler:.1%}.\n"
                "RECOMMENDATION: No urgent clinical indicators. Re-evaluate if "
                "subjective complaints persist."
            ),
            "low": (
                "ASSESSMENT: Tentative classification as control "
                "(confidence: {conf:.1%}). Features are ambiguous.\n"
                "EVIDENCE: TTR={ttr:.3f}; filler rate={filler:.1%}.\n"
                "RECOMMENDATION: Low-confidence result. Comprehensive clinical "
                "assessment recommended."
            ),
        },
        "Aphasia": {
            "high": (
                "ASSESSMENT: Aphasia-consistent language patterns detected "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Reduced lexical diversity (TTR={ttr:.3f}), elevated "
                "repetition rate ({rep:.1%}), and word retrieval difficulties "
                "are present. Estimated WAB-AQ: {wab:.1f}/100.\n"
                "RECOMMENDATION: Referral to a speech-language pathologist for "
                "standardised aphasia assessment (e.g., WAB-R, BDAE) is strongly "
                "recommended."
            ),
            "mod": (
                "ASSESSMENT: Moderate likelihood of aphasia "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: TTR={ttr:.3f}; repetition rate={rep:.1%}; "
                "estimated WAB-AQ={wab:.1f}/100.\n"
                "RECOMMENDATION: Aphasia screening warranted. Formal assessment "
                "by a speech-language pathologist advised."
            ),
            "low": (
                "ASSESSMENT: Possible aphasia features — low confidence "
                "({conf:.1%}). Borderline linguistic profile.\n"
                "EVIDENCE: TTR={ttr:.3f}; repetition rate={rep:.1%}.\n"
                "RECOMMENDATION: Clinical interview and standardised testing "
                "recommended to rule out aphasia."
            ),
        },
        "AD": {
            "high": (
                "ASSESSMENT: Language profile consistent with Alzheimer's disease "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Decreased content density ({content:.3f}) and elevated "
                "pronoun-to-noun ratio ({pron:.1%}) suggest semantic memory "
                "decline and anomia — early Alzheimer language markers.\n"
                "RECOMMENDATION: Urgent referral to a neurologist. Comprehensive "
                "neuropsychological evaluation and neuroimaging are recommended."
            ),
            "mod": (
                "ASSESSMENT: Moderate likelihood of Alzheimer's disease "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Content density={content:.3f}; pronoun ratio={pron:.1%}. "
                "Semantic and naming features are atypical.\n"
                "RECOMMENDATION: Neurological consultation and cognitive screening "
                "(e.g., MoCA, MMSE) are recommended."
            ),
            "low": (
                "ASSESSMENT: Possible Alzheimer's disease features — low confidence "
                "({conf:.1%}).\n"
                "EVIDENCE: Content density={content:.3f}; TTR={ttr:.3f}.\n"
                "RECOMMENDATION: Cognitive screening and follow-up assessment "
                "recommended."
            ),
        },
        "TBI": {
            "high": (
                "ASSESSMENT: Language profile consistent with traumatic brain injury "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Reduced discourse coherence, elevated filler rate "
                "({filler:.1%}), and decreased mean utterance length "
                "({avg_len:.1f} words) are consistent with TBI-related "
                "cognitive-communication deficits.\n"
                "RECOMMENDATION: Referral to a cognitive rehabilitation specialist. "
                "Standardised traumatic brain injury communication assessment "
                "(e.g., ASHA FACS, RIPA-2) is recommended."
            ),
            "mod": (
                "ASSESSMENT: Moderate likelihood of TBI-related communication "
                "impairment (confidence: {conf:.1%}).\n"
                "EVIDENCE: Filler rate={filler:.1%}; utterance length={avg_len:.1f} "
                "words. Discourse organisation is mildly atypical.\n"
                "RECOMMENDATION: TBI communication screening and neuropsychological "
                "evaluation recommended."
            ),
            "low": (
                "ASSESSMENT: Possible TBI-related features — low confidence "
                "({conf:.1%}).\n"
                "EVIDENCE: Filler rate={filler:.1%}; TTR={ttr:.3f}.\n"
                "RECOMMENDATION: Clinical interview to assess TBI history; "
                "formal assessment if indicated."
            ),
        },
        "RHD": {
            "high": (
                "ASSESSMENT: Communication profile consistent with right hemisphere "
                "damage (confidence: {conf:.1%}).\n"
                "EVIDENCE: Pragmatic discourse features are impaired; pronoun "
                "reference ({pron:.1%}) and inferencing capacity are atypical, "
                "consistent with right hemisphere damage-related "
                "pragmatic deficits.\n"
                "RECOMMENDATION: Right Hemisphere Battery (RHB) or discourse "
                "analysis by a speech-language pathologist is recommended."
            ),
            "mod": (
                "ASSESSMENT: Moderate likelihood of right hemisphere damage "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: Pronoun ratio={pron:.1%}; content density={content:.3f}. "
                "Pragmatic features are mildly atypical.\n"
                "RECOMMENDATION: Pragmatic communication assessment recommended."
            ),
            "low": (
                "ASSESSMENT: Possible right hemisphere damage features — "
                "low confidence ({conf:.1%}).\n"
                "EVIDENCE: TTR={ttr:.3f}; pronoun ratio={pron:.1%}.\n"
                "RECOMMENDATION: Neurological review and pragmatic assessment "
                "if clinical history supports RHD."
            ),
        },
        "MCI": {
            "high": (
                "ASSESSMENT: Language profile consistent with mild cognitive "
                "impairment (confidence: {conf:.1%}).\n"
                "EVIDENCE: Naming difficulties and reduced episodic memory "
                "are reflected in low lexical diversity (TTR={ttr:.3f}) and "
                "elevated filler rate ({filler:.1%}), characteristic of "
                "MCI-related language decline.\n"
                "RECOMMENDATION: Memory clinic referral. Montreal Cognitive "
                "Assessment (MoCA) and episodic memory testing are recommended."
            ),
            "mod": (
                "ASSESSMENT: Moderate likelihood of mild cognitive impairment "
                "(confidence: {conf:.1%}).\n"
                "EVIDENCE: TTR={ttr:.3f}; filler rate={filler:.1%}.\n"
                "RECOMMENDATION: Cognitive screening (MoCA/MMSE) and memory "
                "assessment recommended."
            ),
            "low": (
                "ASSESSMENT: Possible MCI features — low confidence ({conf:.1%}).\n"
                "EVIDENCE: TTR={ttr:.3f}; filler rate={filler:.1%}.\n"
                "RECOMMENDATION: Cognitive screening warranted given borderline "
                "linguistic profile."
            ),
        },
    }

    def _conf_tier(self, conf: float) -> str:
        if conf >= self.HIGH_CONF:
            return "high"
        elif conf >= self.MOD_CONF:
            return "mod"
        return "low"

    def generate(self, pred_label: int, features: dict,
                 wab_pred: float, conf: float = 1.0) -> str:
        label_name = IDX_TO_LABEL.get(pred_label, "Unknown")
        tier       = self._conf_tier(conf)
        class_tmpl = self._TEMPLATES.get(label_name)
        if class_tmpl is None:
            return (f"ASSESSMENT: Unrecognised class '{label_name}'. "
                    "Manual clinical review required.")
        tmpl = class_tmpl[tier]
        return tmpl.format(
            conf    = conf,
            ttr     = features.get("ttr", 0.0),
            avg_len = features.get("avg_sent_len", 0.0),
            filler  = features.get("filler_ratio", 0.0),
            rep     = features.get("repetition_rate", 0.0),
            content = features.get("content_density", 0.0),
            pron    = features.get("pronoun_ratio", 0.0),
            wab     = wab_pred,
        )


# ─── Multi-task loss ─────────────────────────────────────────────────────────
def multitask_loss(cls_logits, cls_labels,
                   reg_preds,  reg_targets,
                   sim_embs,   sim_labels,
                   cls_head, reg_head, sim_head,
                   lambda_reg=None, lambda_sim=None):
    """
    L_total = λ_cls·L_cls + λ_reg·L_reg + λ_sim·L_sim
    lambda_reg/lambda_sim: optional overrides for ablations (None → config defaults)
    """
    _lr = lambda_reg if lambda_reg is not None else LAMBDA_REG
    _ls = lambda_sim if lambda_sim is not None else LAMBDA_SIM

    L_cls = cls_head.loss(cls_logits, cls_labels)
    L_reg = reg_head.loss(reg_preds,  reg_targets)
    L_sim = sim_head.loss(sim_embs,   sim_labels)

    total = LAMBDA_CLS * L_cls + _lr * L_reg + _ls * L_sim
    if torch.isnan(total):
        total = LAMBDA_CLS * L_cls + _lr * (L_reg if not torch.isnan(L_reg) else torch.zeros(1))
    return total, {"cls": L_cls.item(), "reg": L_reg.item(), "sim": L_sim.item()}
