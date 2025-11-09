import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  // "/": [
  //   "",
  //   {
  //     text: "DDemo",
  //     icon: "laptop-code",
  //     prefix: "demo/",
  //     link: "demo/",
  //     children: "structure",
  //   },
  //   {
  //     text: "Articles",
  //     icon: "book",
  //     prefix: "posts/",
  //     children: "structure",
  //   },
  //   "intro",
  //   {
  //     text: "Slides",
  //     icon: "person-chalkboard",
  //     link: "https://ecosystem.vuejs.press/plugins/markdown/revealjs/demo.html",
  //   },
  // ],
  "/": ["",],
  "/demo/": ["", {text: "DDemo", icon: "jiaocheng", children: "structure"}],
  "/posts/": ["", {text: "Posts", icon: "blog", children: "structure"}],
});
