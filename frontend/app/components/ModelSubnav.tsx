"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const leagueLinks = {
  nfl: [
    { href: "/models/nfl", label: "Games" },
    { href: "/models/nfl/results", label: "Results" },
    { href: "/models/nfl/how-it-works", label: "How it works" },
    { href: "/models/nfl/insights", label: "Insights" },
  ],
  cfb: [
    { href: "/models/cfb", label: "Games" },
    { href: "/models/cfb/results", label: "Results" },
  ],
} as const;

export function ModelSubnav({ league }: { league: keyof typeof leagueLinks }) {
  const pathname = usePathname();

  return (
    <nav aria-label={`${league.toUpperCase()} model navigation`} className="border-b border-[var(--border)] bg-white">
      <div className="hide-scrollbar mx-auto flex max-w-7xl gap-5 overflow-x-auto px-4 sm:px-6 lg:px-8">
        {leagueLinks[league].map((link) => {
          const active = link.href === `/models/${league}` ? pathname === link.href : pathname.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              aria-current={active ? "page" : undefined}
              className={`relative flex min-h-12 shrink-0 items-center text-sm font-medium ${
                active ? "text-[var(--ink)]" : "text-[var(--muted)] hover:text-[var(--ink)]"
              }`}
            >
              {link.label}
              {active ? <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5 rounded-full bg-[var(--accent)]" /> : null}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
