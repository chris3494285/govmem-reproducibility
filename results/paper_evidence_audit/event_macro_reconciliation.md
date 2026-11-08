# Event-macro reconciliation (B2)

macro = unweighted mean over the three events of a base-weighted per-event effect; each per-event effect is the mean over bases of that base's mean scored-pair delta

## Per-provider event values (percentage points)

### Corruption

| Provider | deletion | temporal | update | macro | pair-weighted mean | micro (all pairs) | base-vs-pair gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| openai | -13.6842 | -35.3057 | -44.3223 | -31.1041 | -32.3643 | -26.5385 | +1.2602 |
| anthropic | -17.3684 | -26.5286 | -40.6593 | -28.1855 | -29.3257 | -25.8654 | +1.1402 |
| deepseek | -23.2456 | -24.6548 | -36.2637 | -28.0547 | -29.4811 | -27.7404 | +1.4264 |
| gemini | -17.8947 | -37.0809 | -40.6593 | -31.8783 | -33.0544 | -28.1731 | +1.1761 |
| kimi | -13.8596 | -23.9645 | -45.0549 | -27.6264 | -29.0291 | -24.7596 | +1.4027 |
| grok | -13.1579 | -19.9211 | -22.3443 | -18.4744 | -19.7867 | -17.5962 | +1.3123 |

### Recovery

| Provider | deletion | temporal | update | macro | pair-weighted mean | micro (all pairs) | base-vs-pair gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| openai | +13.5088 | +35.5030 | +44.3223 | +31.1114 | +32.3904 | +26.4904 | -1.2790 |
| anthropic | +17.0175 | +27.2189 | +40.8425 | +28.3597 | +29.4390 | +25.8173 | -1.0793 |
| deepseek | +22.6316 | +25.0493 | +36.6300 | +28.1036 | +29.5677 | +27.5962 | -1.4641 |
| gemini | +17.7193 | +35.5030 | +40.6593 | +31.2939 | +32.4883 | +27.7885 | -1.1944 |
| kimi | +16.2281 | +26.1341 | +43.4066 | +28.5896 | +30.0305 | +26.0577 | -1.4409 |
| grok | +11.6667 | +18.5404 | +22.5275 | +17.5782 | +18.8431 | +16.5385 | -1.2649 |

## Six-provider reconciliation

| Effect | mean event-macro (pp) | published (pp) | pair-weighted event mean (pp) | review's recomputed value (pp) | method gap (pp) |
|---|---:|---:|---:|---:|---:|
| corruption | -27.5539 | -27.55 | -28.8402 | -28.84 | -1.2863 |
| recovery | +27.5061 | +27.51 | +28.7932 | +28.79 | +1.2871 |

## Resolution of B2

- **corruption**: the published value (-27.55 pp) is the mean of the base-weighted per-event estimand (-27.5539 pp). Re-deriving it from pair-weighted per-event means gives -28.8402 pp, which reproduces the reviewer's independent recomputation (-28.84 pp). The difference (-1.2863 pp) is entirely a weighting difference, not an arithmetic error.
- **recovery**: the published value (+27.51 pp) is the mean of the base-weighted per-event estimand (+27.5061 pp). Re-deriving it from pair-weighted per-event means gives +28.7932 pp, which reproduces the reviewer's independent recomputation (+28.79 pp). The difference (+1.2871 pp) is entirely a weighting difference, not an arithmetic error.

Only the `temporal` event mixes bases with different numbers of scored faults (LongMemEval temporal bases support two operators; OAKS Novel temporal bases support three), so `deletion` and `update` are identical under both weightings and the whole gap is confined to `temporal`.

Reconciliation identity errors: none.
