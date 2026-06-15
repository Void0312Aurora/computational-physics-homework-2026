-- Fix Pandoc section counters for reports that use unnumbered top-level
-- visual groups such as "# I. ... {-}" while still numbering the Problem
-- headings below them.
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
      return pandoc.RawBlock("latex", "\\section*{" .. pandoc.utils.stringify(el.content) .. "}")
    end
    return el
  end

  if el.level > 1 then
    el.level = el.level - 1
  end

  return el
end
