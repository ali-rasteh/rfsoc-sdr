# Graphviz setup & rendering gotchas

## Getting `dot`

Check first: `dot -V`. If it's on PATH, you're done. Otherwise:

- **Linux:** `apt-get install graphviz` (or `dnf`, `brew`, `conda install graphviz`).
- **macOS:** `brew install graphviz`.
- **Windows:** `winget install Graphviz.Graphviz` **frequently reports success but
  leaves no binary on PATH.** The reliable fallback is the portable ZIP:
  ```bash
  # download the win64 zip from the Graphviz GitLab release, extract, use dot.exe directly
  # e.g. .../Graphviz-<ver>-win64/bin/dot.exe
  ```
  After extracting, run `dot.exe -c` once to register the plugin config, then call the
  full path to `dot.exe`. Locate an existing install with
  `Get-ChildItem 'C:\' -Recurse -Filter dot.exe -ErrorAction SilentlyContinue`.

## Rendering

```bash
dot -Tsvg diagram.dot -o diagram.svg     # vector, for the web / editing
dot -Tpdf diagram.dot -o diagram.pdf     # vector, for papers
dot -Tpng -Gdpi=110 diagram.dot -o diagram.png   # raster, for quick visual QA
```

Always eyeball a PNG before declaring done — layout problems (overlaps, sprawl) are
invisible in the DOT source.

## Gotchas that cost real time

1. **Port references must be `"node":"port"`, not `"node:port"`.** If you quote the
   whole thing, Graphviz creates a *new phantom node* literally named `node:port`
   instead of attaching to the HTML-table port. This only bites when you hand-author
   hero blocks with ports; the auto-generator uses plain node ids and sidesteps it.

2. **`splines=ortho` silently drops inline edge `label`s** (you'll see
   `Warning: Orthogonal edges do not currently handle edge labels`). Use `xlabel` +
   `forcelabels=true` instead, which ortho honors.

3. **`pack=false`** for a floating legend node — see visual_conventions.md.

4. **Empty HTML `<font></font>` cells are a parse error.** For a blank table cell emit
   a bare `<td> </td>` (or use `colspan`), never `<td><font point-size="9"></font></td>`.

5. **Escape `&`, `<`, `>` inside HTML-like labels** (`&amp;` etc.) — an unescaped `&`
   in a graph/label produces `not well-formed (invalid token)`.

6. **UTF-8 glyphs:** em dash `—`, en dash `–`, middle dot `·` render fine in the default
   font; fancier arrows (`▸`, `◂`) often show as tofu boxes — prefer edge arrowheads for
   direction and keep labels to plain text.

7. **Wide LR pipelines** (data plane) are legitimately wide landscape figures. Don't
   fight it with layout hacks; export PDF and place it as a full-width figure.
