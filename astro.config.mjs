import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://eesungkim.com/',
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-light',
      wrap: true,
    },
  },
  vite: {
    // Dev server is opened from other machines on the LAN.
    server: { allowedHosts: ['lambda-quad', 'lambda-quad.local'] },
  },
  output: 'static',
  build: {
    format: 'directory'
  }
});
