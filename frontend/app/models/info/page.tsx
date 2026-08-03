/* eslint-disable @next/next/no-img-element */
import Markdown from "react-markdown";
import { Card } from "@/app/components/Card";
import { getModelInfoMarkdown } from "@/app/lib/model-info";

export default async function InfoPage() {
  const readme = await getModelInfoMarkdown();

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-28 lg:pb-24 lg:px-12 text-gray-400">
      <Card>
        <Markdown
          components={{
            h1: ({ children }) => <h1 className="text-3xl font-bold text-gray-200 mb-6">{children}</h1>,
            h2: ({ children }) => <h2 className="text-2xl font-semibold text-gray-300 mt-8 mb-4">{children}</h2>,
            h3: ({ children }) => <h3 className="text-xl font-medium text-gray-300 mt-6 mb-3">{children}</h3>,
            p: ({ children }) => <p className="text-gray-400 mb-4 leading-relaxed">{children}</p>,
            ul: ({ children }) => <ul className="list-disc list-inside mb-4 ml-4 space-y-2">{children}</ul>,
            li: ({ children }) => <li className="text-gray-400">{children}</li>,
            strong: ({ children }) => <strong className="text-gray-300 font-semibold">{children}</strong>,
            img: ({ src, alt }) => <img src={src} alt={alt ?? ""} className="my-6 rounded-lg max-w-full" />,
          }}
        >
          {readme}
        </Markdown>
      </Card>
    </div>
  );
}
