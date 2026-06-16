/**
 * @see https://theme-plume.vuejs.press/config/navigation/ 查看文档了解配置详情
 *
 * Navbar 配置文件，它在 `.vuepress/plume.config.ts` 中被导入。
 */

import { defineNavbarConfig } from "vuepress-theme-plume";

export const enNavbar = defineNavbarConfig([
  { text: "Home", link: "/" },
  { text: "Blog", link: "/blog/" },
  { text: "Project", link: "/project/" },
  { text: "Tags", link: "/article/tags/" },
  { text: "Archives", link: "/article/archives/" },
  { text: "About", link: "/about/" },
  // {
  //   text: "Notes",
  //   items: [{ text: "Demo", link: "/demo/README.md" }],
  // },
]);

export const zhNavbar = defineNavbarConfig([
  { text: "首页", link: "/zh/" },
  { text: "博客", link: "/zh/blog/" },
  { text: "项目", link: "/zh/project/" },
  { text: "标签", link: "/zh/article/tags/" },
  { text: "归档", link: "/zh/article/archives/" },
  { text: "关于", link: "/zh/about/" },
  // {
  //   text: "笔记",
  //   items: [{ text: "示例", link: "/zh/demo/README.md" }],
  // },
]);
