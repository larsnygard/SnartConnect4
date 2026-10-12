# SnartConnect4

Connect 4 — first to line up four wins. Full columns push out the bottom.

- Single self-contained `index.html` — no build step, no dependencies
- 7×6 board, horizontal / vertical / diagonal wins
- Full-column rule: the column shifts down one place, the bottom piece falls out
- One 4-line wins, two 4-lines at once is a draw, connect 5–7 is won by the longest connect
- 2-player mode and vs-CPU mode (minimax with alpha-beta pruning)
- Synthesized Web Audio sound effects with an on/off toggle

## License

Copyright (C) 2026 Lars Christian Nygård (Snart apps).

This project is licensed under the GNU Affero General Public License v3.0-or-later — see [LICENSE](LICENSE).
