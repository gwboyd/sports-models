"use client";

import Link, { useLinkStatus } from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { href: "/models/nfl", pathname: "/models/nfl", label: "NFL" },
  { href: "/models/cfb", pathname: "/models/cfb", label: "CFB" },
  { href: "/models/nba?bankroll=500", pathname: "/models/nba", label: "NBA" },
];

function PendingHint() {
  const { pending } = useLinkStatus();
  return pending ? <span aria-hidden className="ml-2 opacity-60">…</span> : null;
}

export function ModelTabs() {
  const pathname = usePathname();

  return (
    <nav aria-label="League navigation" className="flex items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-100 p-0.5">
      {tabs.map((tab) => {
        const isActive = pathname.startsWith(tab.pathname);
        return (
          <Link
            key={tab.label}
            href={tab.href}
            aria-current={isActive ? "page" : undefined}
            className={`flex min-h-10 items-center justify-center rounded-md px-3.5 py-1.5 text-sm font-semibold transition-colors ${
              isActive
                ? "bg-white text-[var(--ink)] shadow-sm"
                : "text-[var(--muted)] hover:bg-white/70 hover:text-[var(--ink)]"
            }`}
          >
            {tab.label}
            <PendingHint />
          </Link>
        );
      })}
    </nav>
  );
}
