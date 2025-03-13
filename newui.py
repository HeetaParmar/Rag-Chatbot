import { BrowserRouter as Router, Route, Routes, useNavigate } from "react-router-dom";
import { useState } from "react";
import { Button, Input, Card, CardContent } from "@/components/ui";

function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    const response = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password }),
    });

    if (response.ok) {
      navigate("/select-company");
    } else {
      alert("Invalid credentials");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-red-500 via-white to-blue-500">
      <Card className="w-96 p-6 shadow-lg">
        <CardContent>
          <h1 className="text-xl font-bold text-center mb-4">Login</h1>
          <form onSubmit={handleLogin} className="flex flex-col gap-4">
            <Input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <Button type="submit" className="w-full">Login</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

function SelectCompany() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-xl font-bold mb-4">Select a Company</h1>
      <Button onClick={() => navigate("/select-model")}>Company A</Button>
      <Button onClick={() => navigate("/select-model")}>Company B</Button>
    </div>
  );
}

function SelectModel() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-xl font-bold mb-4">Select a Model</h1>
      <Button onClick={() => navigate("/chat")}>Model X</Button>
      <Button onClick={() => navigate("/chat")}>Model Y</Button>
    </div>
  );
}

function Chatbot() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-xl font-bold mb-4">Chatbot Interface</h1>
      <p>Chat functionality will go here.</p>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/select-company" element={<SelectCompany />} />
        <Route path="/select-model" element={<SelectModel />} />
        <Route path="/chat" element={<Chatbot />} />
      </Routes>
    </Router>
  );
}
