(function() {
  const EXTRA_MARKER = "data-sidebar-extra";
  const CONTRIBUTING_LINK = {
    href: "https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md",
    title: "Contributing Guide \u2192",
  };
  const BOTTOM_LINKS = [
    { href: "https://blog.comma.ai", title: "Blog \u2192" },
    { href: "https://comma.ai/bounties", title: "Bounties \u2192" },
    { href: "https://github.com/commaai", title: "GitHub \u2192" },
    { href: "https://discord.comma.ai", title: "Discord \u2192" },
    { href: "https://x.com/comma_ai", title: "X \u2192" },
  ];

  function buildPart(title) {
    const part = document.createElement("li");
    part.className = "part-title sidebar-extra-part";
    part.setAttribute(EXTRA_MARKER, "true");
    part.textContent = title;
    return part;
  }

  function buildLink(link) {
    const item = document.createElement("li");
    item.className = "chapter-item expanded sidebar-extra-link";
    item.setAttribute(EXTRA_MARKER, "true");

    const wrapper = document.createElement("span");
    wrapper.className = "chapter-link-wrapper";

    const anchor = document.createElement("a");
    anchor.href = link.href;
    anchor.textContent = link.title;

    wrapper.append(anchor);
    item.append(wrapper);
    return item;
  }

  function roadmapItem(chapterList) {
    return Array.from(chapterList.querySelectorAll("a")).find((anchor) => anchor.getAttribute("href") === "contributing/roadmap.html")?.closest("li");
  }

  function injectExtras(chapterList) {
    if (!chapterList || chapterList.querySelector(`[${EXTRA_MARKER}]`)) {
      return;
    }

    const roadmap = roadmapItem(chapterList);
    if (roadmap) {
      roadmap.insertAdjacentElement("afterend", buildLink(CONTRIBUTING_LINK));
    }

    chapterList.append(buildPart("Links"));
    for (const link of BOTTOM_LINKS) {
      chapterList.append(buildLink(link));
    }
  }

  function maybeInject() {
    const chapterList = document.querySelector("#mdbook-sidebar .sidebar-scrollbox .chapter");
    if (!chapterList) {
      return false;
    }

    injectExtras(chapterList);
    return true;
  }

  function main() {
    if (maybeInject()) {
      return;
    }

    const sidebar = document.querySelector("#mdbook-sidebar .sidebar-scrollbox");
    if (!sidebar) {
      return;
    }

    const observer = new MutationObserver(() => {
      if (maybeInject()) {
        observer.disconnect();
      }
    });

    observer.observe(sidebar, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main, { once: true });
  } else {
    main();
  }
})();
