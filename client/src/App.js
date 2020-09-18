import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [data, setData] = useState();

  const fetchData = async () => {
    const res = await fetch("http://localhost:5000");
    const json = await res.json();
    setData(json.data);
  };

  useEffect(() => {
    fetchData();
  }, []);

  return <div className="App">{data}</div>;
}

export default App;
