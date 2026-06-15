-- Fix Pandoc section counters for reports that use unnumbered top-level visual
-- groups such as "# I. ... {-}" while still numbering the Problem headings
-- below them as 1.1, 1.2, ...
--
-- Without this filter, --number-sections renders the first numbered level-2
-- heading as 0.1 because its level-1 parent is intentionally unnumbered.

local function has_class(el, class_name)
  for _, class in ipairs(el.classes) do
    if class == class_name then
      return true
    end
  end
  return false
end

function Header(el)
  if el.level == 1 and has_class(el, "unnumbered") then
    if FORMAT:match("latex") then
      local title = pandoc.write(pandoc.Pandoc({ pandoc.Plain(el.content) }), "latex")
      title = title:gsub("%s+$", "")
      return pandoc.RawBlock(
        "latex",
        "\\refstepcounter{section}\\setcounter{subsection}{0}\\setcounter{subsubsection}{0}\\section*{" .. title .. "}"
      )
    end
    return el
  end

  return el
end
