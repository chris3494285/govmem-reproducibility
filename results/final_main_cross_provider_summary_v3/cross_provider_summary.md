# Six-provider final-main cross-provider summary

| Model | Clean | Corrupted | Enforced | Corruption macro | 95% CI | Recovery macro | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Pro | 48.29% | 19.33% | 48.65% | -31.70 pp | [-35.12, -28.39] | +31.65 pp | [28.50, 34.92] |
| GPT-5.6 Sol | 44.87% | 16.92% | 44.66% | -33.67 pp | [-36.72, -30.56] | +33.68 pp | [30.63, 36.73] |
| Claude Sonnet 5 | 45.01% | 17.84% | 45.00% | -31.14 pp | [-34.43, -27.92] | +31.12 pp | [28.17, 34.12] |
| Gemini 3 Flash | 47.33% | 16.30% | 46.88% | -37.11 pp | [-40.26, -33.93] | +37.01 pp | [33.85, 40.08] |
| Kimi K2.6 no-thinking | 46.92% | 19.09% | 47.98% | -33.64 pp | [-36.93, -30.30] | +34.41 pp | [31.28, 37.47] |
| Grok 4.3 no-reasoning | 30.51% | 11.97% | 30.38% | -21.33 pp | [-24.85, -17.85] | +21.42 pp | [18.31, 24.62] |

## Descriptive six-provider aggregates

- Event-macro corruption mean/range: -31.43 pp; -37.11 to -21.33 pp
- Event-macro recovery mean/range: +31.55 pp; +21.42 to +37.01 pp

## Event breakdown

| Model | Deletion corruption/recovery | Temporal corruption/recovery | Update corruption/recovery |
|---|---:|---:|---:|
| DeepSeek V4 Pro | -23.25 / +22.63 | -38.32 / +38.58 | -36.26 / +36.63 |
| GPT-5.6 Sol | -13.68 / +13.51 | -45.69 / +45.94 | -44.32 / +44.32 |
| Claude Sonnet 5 | -17.37 / +17.02 | -37.56 / +37.56 | -40.66 / +40.84 |
| Gemini 3 Flash | -17.89 / +17.72 | -54.06 / +53.81 | -40.66 / +40.66 |
| Kimi K2.6 no-thinking | -13.86 / +16.23 | -43.65 / +45.43 | -45.05 / +43.41 |
| Grok 4.3 no-reasoning | -13.16 / +11.67 | -31.22 / +32.23 | -22.34 / +22.53 |

## Fault breakdown

| Model | Conflicting | Flipped | Missing | Stale |
|---|---:|---:|---:|---:|
| DeepSeek V4 Pro | -5.50 | -37.37 | -35.70 | -50.43 |
| GPT-5.6 Sol | -8.41 | -19.47 | -32.15 | -61.82 |
| Claude Sonnet 5 | -6.15 | -27.11 | -32.83 | -53.28 |
| Gemini 3 Flash | -8.25 | -26.58 | -36.25 | -63.25 |
| Kimi K2.6 no-thinking | -2.59 | -26.58 | -34.34 | -59.26 |
| Grok 4.3 no-reasoning | -5.83 | -19.74 | -21.48 | -36.18 |

## Source breakdown

| Model | LongMemEval | Memora | OAKS |
|---|---:|---:|---:|
| DeepSeek V4 Pro | -21.30 / +19.78 | -23.25 / +22.63 | -52.29 / +54.37 |
| GPT-5.6 Sol | -29.57 / +29.57 | -13.68 / +13.51 | -59.58 / +59.79 |
| Claude Sonnet 5 | -27.61 / +27.17 | -17.37 / +17.02 | -50.62 / +51.25 |
| Gemini 3 Flash | -33.48 / +33.48 | -17.89 / +17.72 | -58.54 / +58.33 |
| Kimi K2.6 no-thinking | -29.57 / +28.70 | -13.86 / +16.23 | -58.75 / +59.17 |
| Grok 4.3 no-reasoning | -2.61 / +4.13 | -13.16 / +11.67 | -48.54 / +48.12 |
