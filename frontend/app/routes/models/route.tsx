import { NavLink, Outlet } from "@remix-run/react";
import type { LoaderFunctionArgs } from "@remix-run/server-runtime";
import { redirect } from "@remix-run/server-runtime";

export const loader = ({ request }: LoaderFunctionArgs) => {
  const { pathname } = new URL(request.url);
  if (pathname === "/models") {
    return redirect("/models/nfl");
  }
  return null;
};

export default function Models() {
  return (
    <div className="flex flex-col overflow-hidden">
      <div className="flex border-gray-700 pt-1 px-4 gap-1">
        <Tab to="nfl">NFL</Tab>
        <Tab to="nba?bankroll=500">NBA</Tab>
      </div>
      <Outlet />
    </div>
  );
}

const Tab = ({ to, children }: { to: string; children: React.ReactNode }) => {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 text-sm font-medium rounded-lg ${
          isActive
            ? "text-white bg-gray-700"
            : "text-gray-300 transition-all hover:text-white hover:bg-gray-800"
        }`
      }
    >
      {children}
    </NavLink>
  );
};
