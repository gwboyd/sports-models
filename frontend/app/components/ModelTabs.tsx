"use client";

import Link, { useLinkStatus } from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { href: "/models/nfl", pathname: "/models/nfl", label: "NFL" },
  { href: "/models/nba?bankroll=500", pathname: "/models/nba", label: "NBA" },
  { href: "/models/info", pathname: "/models/info", label: "Info" },
];

function PendingHint() {
  const { pending } = useLinkStatus();
  return pending ? <span aria-hidden className="ml-2 opacity-60">…</span> : null;
}

export function ModelTabs() {
  const pathname = usePathname();

  return (
    <nav aria-label="Model navigation" className="flex border-gray-700 pt-1 px-4 gap-1">
      {tabs.map((tab) => {
        const isActive = pathname === tab.pathname;
        return (
          <Link
            key={tab.label}
            href={tab.href}
            aria-current={isActive ? "page" : undefined}
            className={`px-4 py-2 text-sm font-medium rounded-lg ${
              isActive
                ? "text-white bg-gray-700"
                : "text-gray-300 transition-all hover:text-white hover:bg-gray-800"
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
