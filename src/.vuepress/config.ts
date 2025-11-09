import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "en-US",
  title: "Kstheme's Blog",
  description: "I never think of the future. It comes soon enough",

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
