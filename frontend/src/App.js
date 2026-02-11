import './App.css';
import PredictionForm from './PredictionForm';

function App() {
  return (
    <div className="App">
       <header className="App-header">
        <h1>AI Skin Lesion Analyzer</h1>
      </header>
      <main>
        <PredictionForm />
      </main>
      <footer>
        <p>Important: This tool is for informational purposes only and not a medical diagnosis.</p>
      </footer>
    </div
    >
  );
}

export default App;
