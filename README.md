# Rhythmic Epistles

> 欲買桂花同載酒，終不似，少年遊。

Personal blog powered by [Jekyll](https://jekyllrb.com/), hosted on [GitHub Pages](https://pages.github.com/).

**Live site:** [kakack.github.io](https://kakack.github.io)

---

## Features

- **Theme:** Clean sidebar layout, teal accent; dark mode toggle
- **Content:** Markdown posts, tags, archive, optional TOC per post
- **Math:** LaTeX (MathJax) for inline `$...$` and display `$$...$$`
- **Code:** Syntax highlighting (Rouge), copy-to-clipboard on code blocks
- **Reading:** Progress bar and back-to-top on article pages
- **Comments:** [Utterances](https://utteranc.es/) (GitHub-based)
- **Feed:** Atom feed at `/feed.xml`

---

## Local development

Requires **Ruby ≥ 3.0** (e.g. `rbenv install 3.2.0` or `brew install ruby`).

```bash
git clone https://github.com/kakack/kakack.github.io.git
cd kakack.github.io
bundle install
bundle exec jekyll serve
```

Open [http://127.0.0.1:4000](http://127.0.0.1:4000). Use `--host 0.0.0.0` to serve on LAN.

---

## Project structure

```
├── _config.yml       # Site config, nav, plugins
├── _includes/        # Head, nav, footer, comments, etc.
├── _layouts/         # default, post, page
├── _posts/           # Markdown articles
├── _sass/            # SCSS (variables, dark mode, highlights)
├── style.scss        # Main stylesheet entry
├── index.html        # Home + pagination
├── about.md          # About page
├── archive/          # Archive index
└── tags/             # Tag index
```

---

## License

[MIT](LICENSE).

*Let's talk with the world.*
