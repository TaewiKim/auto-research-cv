"""Clinical questionnaire and scale parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScaleType(str, Enum):
    """Types of clinical scales."""

    MOTOR = "motor"
    COGNITIVE = "cognitive"
    FUNCTIONAL = "functional"
    QUALITY_OF_LIFE = "quality_of_life"
    COMPOSITE = "composite"


@dataclass
class ScaleItem:
    """Individual item in a clinical scale."""

    number: str
    name: str
    description: str
    min_score: int
    max_score: int
    score_descriptions: dict[int, str] = field(default_factory=dict)


@dataclass
class ClinicalScale:
    """Clinical assessment scale."""

    name: str
    abbreviation: str
    scale_type: ScaleType
    description: str
    items: list[ScaleItem]
    total_min: int
    total_max: int
    reference: str | None = None

    @property
    def item_count(self) -> int:
        return len(self.items)


# MDS-UPDRS Part III (Motor Examination)
MDS_UPDRS_PART3 = ClinicalScale(
    name="Movement Disorder Society - Unified Parkinson's Disease Rating Scale Part III",
    abbreviation="MDS-UPDRS-III",
    scale_type=ScaleType.MOTOR,
    description="Motor examination section of the MDS-UPDRS for Parkinson's disease",
    reference="Goetz et al., 2008",
    total_min=0,
    total_max=132,
    items=[
        ScaleItem(
            number="3.1",
            name="Speech",
            description="Spontaneous speech assessment",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight loss of expression, diction, and/or volume",
                2: "Moderate impairment",
                3: "Severe impairment",
                4: "Unintelligible",
            },
        ),
        ScaleItem(
            number="3.2",
            name="Facial Expression",
            description="Hypomimia assessment",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Minimal hypomimia",
                2: "Slight but definite abnormal diminution",
                3: "Moderate hypomimia",
                4: "Masked or fixed facies",
            },
        ),
        ScaleItem(
            number="3.3",
            name="Rigidity",
            description="Judged on passive movement of major joints",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Absent",
                1: "Slight",
                2: "Mild to moderate",
                3: "Marked",
                4: "Severe",
            },
        ),
        ScaleItem(
            number="3.4",
            name="Finger Tapping",
            description="Rapid alternating movements of index finger",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight slowing and/or reduction in amplitude",
                2: "Mild impairment",
                3: "Moderate impairment",
                4: "Severe impairment",
            },
        ),
        ScaleItem(
            number="3.5",
            name="Hand Movements",
            description="Open and close hands in rapid succession",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight slowing and/or reduction in amplitude",
                2: "Mild impairment",
                3: "Moderate impairment",
                4: "Severe impairment",
            },
        ),
        ScaleItem(
            number="3.6",
            name="Pronation-Supination",
            description="Rapid alternating movements of hands",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight",
                2: "Mild",
                3: "Moderate",
                4: "Severe",
            },
        ),
        ScaleItem(
            number="3.7",
            name="Toe Tapping",
            description="Rapid alternating toe tapping",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight",
                2: "Mild",
                3: "Moderate",
                4: "Severe",
            },
        ),
        ScaleItem(
            number="3.8",
            name="Leg Agility",
            description="Raise and stomp feet on ground",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight",
                2: "Mild",
                3: "Moderate",
                4: "Severe",
            },
        ),
        ScaleItem(
            number="3.9",
            name="Arising from Chair",
            description="Rising from straight-backed chair with arms folded",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slow, may need multiple attempts",
                2: "Pushes self up from arms of seat",
                3: "Tends to fall back, may have to try more than once",
                4: "Unable to arise without help",
            },
        ),
        ScaleItem(
            number="3.10",
            name="Gait",
            description="Walking assessment",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Minor disturbance",
                2: "Walks with some difficulty",
                3: "Severe disturbance, requires assistance",
                4: "Cannot walk at all",
            },
        ),
        ScaleItem(
            number="3.11",
            name="Freezing of Gait",
            description="Assessment of gait freezing episodes",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Start hesitation",
                2: "Occasional freezing",
                3: "Frequent freezing",
                4: "Frequent falls from freezing",
            },
        ),
        ScaleItem(
            number="3.12",
            name="Postural Stability",
            description="Response to sudden displacement",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Retropulsion with recovery unaided",
                2: "Would fall if not caught",
                3: "Very unstable, loses balance spontaneously",
                4: "Unable to stand without assistance",
            },
        ),
        ScaleItem(
            number="3.13",
            name="Posture",
            description="Postural alignment assessment",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight stooping",
                2: "Moderate stooping",
                3: "Severe stooping with kyphosis",
                4: "Marked flexion with extreme abnormality",
            },
        ),
        ScaleItem(
            number="3.14",
            name="Global Spontaneity of Movement",
            description="Overall bradykinesia",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Normal",
                1: "Slight global slowness",
                2: "Mild global slowness",
                3: "Moderate global slowness",
                4: "Severe global slowness",
            },
        ),
        ScaleItem(
            number="3.15",
            name="Postural Tremor",
            description="Tremor with hands extended",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Absent",
                1: "Slight",
                2: "Moderate amplitude",
                3: "Large amplitude",
                4: "Severe amplitude",
            },
        ),
        ScaleItem(
            number="3.16",
            name="Kinetic Tremor",
            description="Tremor during movement",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Absent",
                1: "Slight",
                2: "Moderate amplitude",
                3: "Large amplitude",
                4: "Severe amplitude",
            },
        ),
        ScaleItem(
            number="3.17",
            name="Rest Tremor Amplitude",
            description="Tremor at rest",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Absent",
                1: "Slight",
                2: "Mild amplitude",
                3: "Moderate amplitude",
                4: "Severe amplitude",
            },
        ),
        ScaleItem(
            number="3.18",
            name="Constancy of Rest Tremor",
            description="Consistency of rest tremor",
            min_score=0,
            max_score=4,
            score_descriptions={
                0: "Absent",
                1: "Slight and infrequent",
                2: "Intermittent",
                3: "Present most of time",
                4: "Present all of time",
            },
        ),
    ],
)

# Hoehn and Yahr Scale
HOEHN_YAHR = ClinicalScale(
    name="Hoehn and Yahr Scale",
    abbreviation="H&Y",
    scale_type=ScaleType.COMPOSITE,
    description="Staging scale for Parkinson's disease severity",
    reference="Hoehn and Yahr, 1967",
    total_min=0,
    total_max=5,
    items=[
        ScaleItem(
            number="1",
            name="Stage",
            description="Overall disease stage",
            min_score=0,
            max_score=5,
            score_descriptions={
                0: "No signs of disease",
                1: "Unilateral involvement only",
                2: "Bilateral involvement without impairment of balance",
                3: "Mild to moderate involvement; some postural instability",
                4: "Severe disability; still able to walk or stand unassisted",
                5: "Wheelchair bound or bedridden unless aided",
            },
        ),
    ],
)

# House-Brackmann Facial Nerve Grading Scale
HOUSE_BRACKMANN = ClinicalScale(
    name="House-Brackmann Facial Nerve Grading System",
    abbreviation="HB",
    scale_type=ScaleType.MOTOR,
    description="Grading system for facial nerve dysfunction",
    reference="House and Brackmann, 1985",
    total_min=1,
    total_max=6,
    items=[
        ScaleItem(
            number="1",
            name="Grade",
            description="Facial nerve function grade",
            min_score=1,
            max_score=6,
            score_descriptions={
                1: "Normal - Normal facial function in all areas",
                2: "Mild dysfunction - Slight weakness noticeable on close inspection",
                3: "Moderate dysfunction - Obvious but not disfiguring difference",
                4: "Moderately severe dysfunction - Obvious weakness and/or disfiguring asymmetry",
                5: "Severe dysfunction - Only barely perceptible motion",
                6: "Total paralysis - No movement",
            },
        ),
    ],
)

# Registry of all scales
SCALE_REGISTRY: dict[str, ClinicalScale] = {
    "mds-updrs-iii": MDS_UPDRS_PART3,
    "hoehn-yahr": HOEHN_YAHR,
    "house-brackmann": HOUSE_BRACKMANN,
}


def get_scale(name: str) -> ClinicalScale | None:
    """Get clinical scale by name or abbreviation."""
    name_lower = name.lower().replace(" ", "-")

    # Direct lookup
    if name_lower in SCALE_REGISTRY:
        return SCALE_REGISTRY[name_lower]

    # Search by abbreviation
    for scale in SCALE_REGISTRY.values():
        if scale.abbreviation.lower() == name_lower:
            return scale

    return None


def list_scales() -> list[ClinicalScale]:
    """List all available clinical scales."""
    return list(SCALE_REGISTRY.values())


def format_scale(scale: ClinicalScale) -> str:
    """Format clinical scale as readable text."""
    lines = [
        f"# {scale.name} ({scale.abbreviation})",
        f"\n{scale.description}",
        f"\nType: {scale.scale_type.value}",
        f"Score Range: {scale.total_min} - {scale.total_max}",
        f"Items: {scale.item_count}",
    ]

    if scale.reference:
        lines.append(f"Reference: {scale.reference}")

    lines.append("\n## Items\n")

    for item in scale.items:
        lines.append(f"### {item.number}. {item.name}")
        lines.append(f"{item.description}")
        lines.append(f"Score: {item.min_score} - {item.max_score}")

        if item.score_descriptions:
            lines.append("\nScoring:")
            for score, desc in sorted(item.score_descriptions.items()):
                lines.append(f"  {score}: {desc}")

        lines.append("")

    return "\n".join(lines)
