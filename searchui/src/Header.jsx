import React from "react";
import { IoMenu } from "react-icons/io5";

export const Header = (props) => {
  const { user } = props;
  return (
    <div className="header-container">
      {/* { Logo} */}
      <div className="header-flex-container">
        <div className="header-text">
          <span>Wells Fargo Compass</span>
        </div>
        <div className="header-top-right-container">
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
