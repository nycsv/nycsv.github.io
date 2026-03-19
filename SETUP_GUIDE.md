# 웹사이트 구현 버전 비교

Hugo 종속성을 제거하고 두 가지 구현 방식을 비교할 수 있도록 준비했습니다.

## 📋 개요

| 항목 | 11ty | Astro |
|------|------|-------|
| **특징** | 경량, 매우 유연 | 현대적, SEO 최적화 |
| **학습곡선** | 낮음 | 중간 |
| **빌드 속도** | 빠름 | 빠름 |
| **수식 지원** | ✅ (markdown-it-mathjax3) | ✅ (rehype-katex) |
| **이미지 최적화** | ✅ (@11ty/eleventy-img) | ✅ (built-in) |
| **다크모드** | ✅ CSS 기반 | ✅ CSS 기반 |
| **마크다운 기능** | 완전 지원 | 완전 지원 |

## 🚀 설치 및 실행

### 공통 요구사항
- **Node.js** 18.x 이상
- **npm** 또는 **yarn**

### 11ty 버전 (eleventy-version 브랜치)

```bash
# 브랜치 전환
git checkout eleventy-version

# 의존성 설치
npm install

# 개발 서버 실행 (http://localhost:8080)
npm run serve

# 빌드
npm run build

# 결과 확인
# 생성 파일: _site/ 디렉토리
```

**특징:**
- `_includes/layouts/` 에서 Nunjucks 템플릿 커스터마이징 가능
- `.eleventy.js` 에서 플러그인 추가 간단
- 빌드 결과물이 `_site/` 에 생성됨

### Astro 버전 (astro-version 브랜치)

```bash
# 브랜치 전환
git checkout astro-version

# 의존성 설치
npm install

# 개발 서버 실행 (http://localhost:3000)
npm run dev

# 빌드
npm run build

# 미리보기
npm run preview

# 결과 확인
# 생성 파일: dist/ 디렉토리
```

**특징:**
- `src/pages/` 에서 파일 기반 라우팅
- `src/content/` 에서 Content Collections 활용
- `.astro` 파일로 컴포넌트 기반 구조
- `astro.config.mjs` 에서 수식 플러그인 설정

## 📁 디렉토리 구조 비교

### 11ty 버전
```
.
├── _includes/
│   └── layouts/         # Nunjucks 템플릿
├── _site/              # 빌드 결과물
├── content/
│   └── en/
│       ├── posts/
│       ├── reviews/
│       └── about.md
├── assets/             # 정적 자산
├── css/               # 스타일시트
├── .eleventy.js       # 11ty 설정
└── index.md           # 홈페이지
```

### Astro 버전
```
.
├── src/
│   ├── content/        # Content Collections
│   │   ├── posts/
│   │   └── reviews/
│   ├── layouts/        # Astro 컴포넌트
│   ├── pages/          # 파일 기반 라우팅
│   ├── components/     # 재사용 컴포넌트
│   └── config.ts       # Content 설정
├── public/
│   └── css/           # 정적 자산
├── dist/              # 빌드 결과물
├── astro.config.mjs   # Astro 설정
└── package.json
```

## ✨ 공통 기능

### 마크다운 지원
- ✅ **LaTeX 수식**: `$$E = mc^2$$`
- ✅ **코드 하이라이팅**: Syntax highlighting
- ✅ **이미지 최적화**: WebP 변환, Lazy loading
- ✅ **제목 앵커**: 자동 생성된 링크
- ✅ **메타 정보**: 작성일, 태그 표시

### 프론트매터 필드
```yaml
---
title: "포스트 제목"
date: 2026-03-18
tags: ["태그1", "태그2"]
---
```

### 자동화된 배포
- **11ty**: `.github/workflows/eleventy-build.yml`
  - 브랜치: `eleventy-version`
  - 결과 배포: `eleventy/` 폴더

- **Astro**: `.github/workflows/astro-build.yml`
  - 브랜치: `astro-version`
  - 결과 배포: `astro/` 폴더

## 🎨 커스터마이징

### 11ty - 스타일 변경
```
css/style.css 수정 후 npm run build
```

### Astro - 레이아웃 변경
```
src/layouts/BaseLayout.astro 수정
src/layouts/PostLayout.astro 수정
```

## 📊 성능 및 선택 기준

### 11ty를 선택하면 좋은 경우
- ✅ 간단한 구조를 빠르게 구축하고 싶을 때
- ✅ 템플릿 언어에 자유도가 필요할 때
- ✅ 가벼운 빌드 프로세스 원할 때

### Astro를 선택하면 좋은 경우
- ✅ 컴포넌트 기반 구조 선호할 때
- ✅ Content Collections로 체계적 관리하고 싶을 때
- ✅ 향후 상호작용 기능(islands) 추가할 가능성이 있을 때

## 🔄 두 버전 비교 실행

```bash
# 11ty 버전 빌드 & 확인
git checkout eleventy-version
npm install && npm run build
# _site/ 폴더에서 확인

# Astro 버전 빌드 & 확인
git checkout astro-version
npm install && npm run build
# dist/ 폴더에서 확인

# 개발 서버로 비교
npm run serve  # 11ty
npm run dev    # Astro
```

## 🎯 최종 선택

**추천: 11ty** (더 간단하고 통제하기 쉬움)
- 구조가 직관적
- 빌드 설정이 명확
- 마크다운 중심 workflow에 최적화

다만 **Astro**도 현대적 구조와 좋은 개발 경험을 제공합니다.
