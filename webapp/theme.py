"""Design tokens - the UX spec's machine-validated palette. Charts and components read
ONLY from here; no literal colors anywhere else in webapp/."""
TOKENS = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#5f5e58",
        "muted": "#898781", "grid": "#e1e0d9", "s1": "#2a78d6", "demph": "#c3c2b7",
        "seq": ["#e8f0fb", "#c4d9f4", "#9dc0ec", "#6fa3e2", "#4488d9", "#2a78d6", "#1c5cab"],
        "div_pos": "#2a78d6", "div_neg": "#d03b3b",
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#b5b3ac",
        "muted": "#898781", "grid": "#2c2c2a", "s1": "#3987e5", "demph": "#52514e",
        "seq": ["#12203a", "#1b3a66", "#255492", "#2f6ebd", "#3987e5", "#61a0ea", "#8ab9f0"],
        "div_pos": "#3987e5", "div_neg": "#e66767",
        "good": "#0ca30c", "warning": "#c98500", "serious": "#ec835a", "critical": "#e66767",
    },
}
STATUS = {
    "healthy": ("✓", "Signal live", "good"),
    "weak": ("◐", "Signal weak", "warning"),
    "degraded": ("⚠", "Signal degraded", "critical"),
    "retired": ("✕", "Signal retired", "muted"),
}
DISCLAIMER = "Educational analytics — not investment advice."
PROBABILITY_SENTENCE = "chance this fund falls below its peers' median return next quarter"
