"""
display_mode_module.py
GUIモード・テーマ・レイアウト・アニメーション設定の定義ファイル。

責務:
  - AppMode / CharacterState の定義
  - カラーパレット（Theme）の定義
  - レイアウト設定（LayoutConfig）の定義
  - アニメーション設定（AnimationConfig）の定義

制約:
  - 表示定義のみ。ロジックを持たない
  - 外部モジュールへの依存なし
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


# ============================================================
# モード定義
# ============================================================

class AppMode(Enum):
    DESKTOP   = "desktop"    # フルUI
    CHARACTER = "character"  # キャラ中心の小ウィンドウ


class CharacterState(Enum):
    IDLE     = "idle"
    SPEAKING = "speaking"
    THINKING = "thinking"
    GREETING = "greeting"
    DEFAULT  = "default"


# ============================================================
# カラーパレット
# ============================================================

@dataclass(frozen=True)
class ColorPalette:
    bg_primary:        str
    bg_secondary:      str
    bg_tertiary:       str
    bg_hover:          str
    text_primary:      str
    text_secondary:    str
    text_muted:        str
    accent_primary:    str
    accent_secondary:  str
    accent_success:    str
    accent_warning:    str
    accent_error:      str
    bubble_user:       str
    bubble_alice:      str
    bubble_user_text:  str
    bubble_alice_text: str
    border:            str
    border_focus:      str
    scrollbar:         str
    scrollbar_hover:   str


class Theme:
    DARK = ColorPalette(
        bg_primary       = "#0d0d1a",
        bg_secondary     = "#13132b",
        bg_tertiary      = "#1a1a35",
        bg_hover         = "#252550",
        text_primary     = "#e8e8ff",
        text_secondary   = "#a0a0cc",
        text_muted       = "#606080",
        accent_primary   = "#7c6fe0",
        accent_secondary = "#5b9cf6",
        accent_success   = "#4ade80",
        accent_warning   = "#fbbf24",
        accent_error     = "#f87171",
        bubble_user      = "#3d2d80",
        bubble_alice     = "#1e2d50",
        bubble_user_text = "#e8e8ff",
        bubble_alice_text= "#d0d8ff",
        border           = "#2a2a55",
        border_focus     = "#7c6fe0",
        scrollbar        = "#2a2a55",
        scrollbar_hover  = "#7c6fe0",
    )

    LIGHT = ColorPalette(
        bg_primary       = "#f0f2ff",
        bg_secondary     = "#e4e8ff",
        bg_tertiary      = "#ffffff",
        bg_hover         = "#d8dcff",
        text_primary     = "#1a1a3e",
        text_secondary   = "#4a4a8a",
        text_muted       = "#9090c0",
        accent_primary   = "#6655d0",
        accent_secondary = "#4488e8",
        accent_success   = "#22aa55",
        accent_warning   = "#cc8800",
        accent_error     = "#cc3333",
        bubble_user      = "#7c6fe0",
        bubble_alice     = "#dde4ff",
        bubble_user_text = "#ffffff",
        bubble_alice_text= "#1a1a3e",
        border           = "#c0c8ff",
        border_focus     = "#6655d0",
        scrollbar        = "#c0c8ff",
        scrollbar_hover  = "#6655d0",
    )

    @classmethod
    def get(cls, name: str) -> ColorPalette:
        return cls.DARK if name == "dark" else cls.LIGHT


# ============================================================
# レイアウト設定
# ============================================================

@dataclass
class LayoutConfig:
    min_width:       int
    min_height:      int
    default_width:   int
    default_height:  int
    resizable:       bool = True
    always_on_top:   bool = False
    show_sidebar:    bool = True
    show_chat_panel: bool = True
    show_character:  bool = True
    show_status_bar: bool = True
    show_toolbar:    bool = True
    char_panel_width:int = 420
    char_size:       Tuple[int, int] = (360, 640)
    chat_font_size:  int = 13
    input_height:    int = 70


LAYOUT_DESKTOP = LayoutConfig(
    min_width=900, min_height=600,
    default_width=1280, default_height=800,
    char_panel_width=540, char_size=(512, 512),
)

LAYOUT_CHARACTER = LayoutConfig(
    min_width=350, min_height=550,
    default_width=380, default_height=680,
    always_on_top=True,
    show_sidebar=False,
    show_status_bar=False,
    show_toolbar=False,
    char_panel_width=540, char_size=(512, 512),
    chat_font_size=12, input_height=60,
)


def get_layout(mode: AppMode) -> LayoutConfig:
    return LAYOUT_DESKTOP if mode == AppMode.DESKTOP else LAYOUT_CHARACTER


# ============================================================
# アニメーション設定
# ============================================================

@dataclass
class AnimationConfig:
    breath_amplitude:   float = 8.0
    breath_period_ms:   int   = 3000
    blink_interval_ms:  int   = 4000
    blink_duration_ms:  int   = 150
    speak_bounce_amp:   float = 4.0
    speak_bounce_period_ms: int = 400
    fade_in_duration_ms: int  = 800
    fps:                int   = 30
    frame_interval_ms:  int   = field(init=False)

    def __post_init__(self) -> None:
        self.frame_interval_ms = max(16, 1000 // self.fps)


DEFAULT_ANIMATION = AnimationConfig()
