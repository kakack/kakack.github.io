<div align="center">
  <img src="https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/logo.png" width="120" alt="logo" />

  <h1>Rhythmic Epistles</h1>

  <p><strong>Where scaling laws meet the open road</strong></p>

  <p>
    <a href="https://kakack.github.io">Live Site</a> ·
    <a href="https://kakack.github.io/archive">Archive</a> ·
    <a href="https://kakack.github.io/tags">Tags</a>
  </p>

  <p>
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/kakack/kakack.github.io?style=flat-square" />
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/kakack/kakack.github.io?style=flat-square" />
    <img alt="License" src="https://img.shields.io/github/license/kakack/kakack.github.io?style=flat-square" />
  </p>
</div>

---

Personal technical blog by **Kyrie Chen** — covering LLM systems, inference optimization, distributed training, AI agents, and the occasional travel essay. 55+ posts and counting since 2014.

Built with [Jekyll](https://jekyllrb.com/) and hosted on [GitHub Pages](https://pages.github.com/).

## Topics

| Category | What you'll find |
|---|---|
| **LLM & Training** | Attention variants, MoE architectures, Scaling Laws, distributed training (DeepSpeed / Megatron-LM) |
| **Inference & Infra** | vLLM internals, KV Cache deep dives, Speculative Decoding, GPU fundamentals |
| **Agent & Application** | Agent design patterns, RAG pipelines, Memory systems, MCP protocol, prompt engineering |
| **Travel & Life** | Long-form essays from Japan, Italy, Spain, Southeast Asia, and across China |

## Features

- Dark / light mode toggle with system preference detection
- Per-post table of contents with scroll-aware highlighting
- LaTeX math rendering (MathJax) — inline `$...$` and display `$$...$$`
- Syntax highlighting (Rouge) with copy-to-clipboard and collapsible long blocks
- Reading progress bar and back-to-top button on article pages
- GitHub-based comments via [Utterances](https://utteranc.es/)
- Atom feed at `/feed.xml`, sitemap at `/sitemap.xml`
- Responsive sidebar that collapses on post pages for wider reading area

## Quick Start

Requires **Ruby ≥ 3.0** and **Bundler**.

```bash
git clone https://github.com/kakack/kakack.github.io.git
cd kakack.github.io
bundle install
bundle exec jekyll serve
```

Open [http://127.0.0.1:4000](http://127.0.0.1:4000). Add `--host 0.0.0.0` to serve on LAN.

## Project Structure

```
kakack.github.io/
├── _config.yml          # Site metadata, navigation, plugins
├── _includes/           # Partials — head, nav, footer, comments, toc
├── _layouts/            # Templates — default, post, page
├── _posts/              # Markdown articles (YYYY-M-D-title.md)
├── _sass/               # SCSS partials — variables, dark mode, highlights
├── style.scss           # Main stylesheet entry point
├── index.html           # Homepage — hero, skills, focus areas, blog feed
├── about.md             # About page
├── archive/             # Chronological archive
└── tags/                # Tag index
```

## Tech Stack

Jekyll · Kramdown · SCSS · MathJax · Rouge · Utterances · GitHub Pages

## License

[MIT](LICENSE)

---

<sub>*Let's talk with the world.*</sub>
