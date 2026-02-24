import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "@/pages/LandingPage";
import EvaluationPage from "@/pages/EvaluationPage";
import StepEvaluationPage from "@/pages/StepEvaluationPage";
import "./index.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/eval" element={<EvaluationPage />} />
        <Route path="/step-eval" element={<StepEvaluationPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
