import React from "react"
import './App.css';
import Sidebar from "./components/Sidebar/Sidebar"
import Main from "./components/Main/Main"

const App = () => {
  return (
    // <>
    //  <Sidebar/>
    //  <Main/>
    // </>
    <div style={{ display: 'flex', height: '100vh', margin: 0, fontFamily: 'Poppins, sans-serif' }}>
    {/* Sidebar */}
    <Sidebar/>
    
    {/* Main content */}

    <Main/>


    {/* <div className="main">
      <div className="main-header">
        <h2>ExaGuru</h2>
        <div className="header-buttons">
          <button><span className="material-icons">add</span></button>
          <button><span className="material-icons">save</span></button>
          <button><span className="material-icons">history</span></button>
        </div>
      </div>
      <div className="content-area">
      </div>
      <div className="chat-input">
        <span className="material-icons">account_circle</span>
        <input type="text" placeholder="Ask ExaGuru Anything..." />
        <button><span className="material-icons">send</span></button>
      </div>
    </div> */}

    


  </div>



  )
}
export default App
