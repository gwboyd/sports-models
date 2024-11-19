import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import { Card } from "~/components/Card";
import Markdown from "react-markdown";
import readme from "../../../../src/nfl/nfl_expected_points/README.md?raw";

export const loader = async () => {
  return json({ readme });
};

export default function Info() {
  const { readme } = useLoaderData<typeof loader>();

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-28 lg:pb-24 lg:px-12 text-gray-400">
      <Card>
        <Markdown
          components={{
            h1: ({ children }) => (
              <h1 className="text-3xl font-bold text-gray-200 mb-6">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-2xl font-semibold text-gray-300 mt-8 mb-4">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-xl font-medium text-gray-300 mt-6 mb-3">
                {children}
              </h3>
            ),
            p: ({ children }) => (
              <p className="text-gray-400 mb-4 leading-relaxed">{children}</p>
            ),
            ul: ({ children }) => (
              <ul className="list-disc list-inside mb-4 ml-4 space-y-2">
                {children}
              </ul>
            ),
            li: ({ children }) => <li className="text-gray-400">{children}</li>,
            strong: ({ children }) => (
              <strong className="text-gray-300 font-semibold">
                {children}
              </strong>
            ),
            img: ({ src, alt }) => (
              <div className="my-6">
                <img src={src} alt={alt} className="rounded-lg max-w-full" />
              </div>
            ),
          }}
        >
          {readme}
        </Markdown>
      </Card>
    </div>
  );
}
