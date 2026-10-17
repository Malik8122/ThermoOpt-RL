import "@fontsource-variable/sora";
import "@fontsource-variable/manrope";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { Compare } from "./pages/Compare";
import { Forecast } from "./pages/Forecast";
import { Overview } from "./pages/Overview";
import { Training } from "./pages/Training";
import { Twin } from "./pages/Twin";
import { Workloads } from "./pages/Workloads";

createRoot(document.getElementById("root")!).render(
  <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Overview />} />
          <Route path="twin" element={<Twin />} />
          <Route path="workloads" element={<Workloads />} />
          <Route path="training" element={<Training />} />
          <Route path="compare" element={<Compare />} />
          <Route path="forecast" element={<Forecast />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
  </BrowserRouter>,
);
