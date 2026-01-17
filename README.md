# Hugo + PaperMod (Bilingual: KO/EN) + GitHub Pages Template

This template sets up a bilingual Hugo blog using the PaperMod theme and deploys it to GitHub Pages via GitHub Actions.

## 1) Prerequisites
- Hugo (extended) installed locally (optional but recommended for local preview)
- Git installed

## 2) Install PaperMod theme (recommended: git submodule)
From the repository root:

```bash
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive
```

## 3) Configure `baseURL`
Edit `hugo.yaml`:

- If you use `username.github.io` repo: `https://username.github.io/`
- If you use a project repo (served under `/repo-name/`): set `baseURL` accordingly, or rely on the workflow's `--baseURL` override.

## 4) Local run
```bash
hugo server -D
```

- Korean home: http://localhost:1313/
- English home: http://localhost:1313/en/

## 5) Enable GitHub Pages (one-time)
Repo Settings → Pages → Source: **GitHub Actions**

Then push to `main`. The workflow `.github/workflows/gh-pages.yaml` will deploy automatically.

## Content structure
- `content/en/...` → English (default, root `/`)
- `content/ko/...` → Korean (`/ko/`)

## Notes
- This template does NOT vendor the theme itself; you should add PaperMod as a submodule (recommended) or copy it into `themes/PaperMod`.
