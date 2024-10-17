import React from "react";
import { IoMenu } from "react-icons/io5";
import { Link } from "react-router-dom";

export const Header = (props) => {
  const { user, fromLandingPage } = props;
  return (
    <div className="header-container">
      {/* { Logo} */}
      <div className="header-flex-container">
        <div className="header-text">
          <span>Wells Fargo Compass</span>
        </div>
        <div className="header-top-right-container">
          {fromLandingPage && (
            <div className="upload-data-link">
              <Link to="/upload">Use your Data</Link>
            </div>
          )}
          <div className="user-image">
            <img src={user.img}></img>
          </div>
          <div>
            <IoMenu color={"#818589"} size={25} />
          </div>
        </div>
      </div>
    </div>
  );
};
