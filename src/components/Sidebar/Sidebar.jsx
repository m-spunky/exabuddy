  import React from "react"
  import {assets} from '../../assets/assets'

  const Sidebar = () => {    
    return (
      <div className="sidebar">
      <h1>EXAGRAM</h1>
      <ul>
        <li><a href="#">Profile</a></li>
        <li><a href="#">ExaBuddy</a></li>
        <li><a href="#">TestGen</a></li>
        <li><a href="#">PYQs</a></li>
        <li><a href="#">ExaConnect</a></li>
        <li><a href="#">ExaGuru</a></li>
        <li><a href="#">ExaVault</a></li>
        <li><a href="#">More</a></li>
      </ul>
      <div className="settings">
        <a href="#">Settings</a>
      </div>
      <a href="#">Logout</a>
    </div>
    )
  }
  export default Sidebar
