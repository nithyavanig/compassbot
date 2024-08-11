const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: path.join(__dirname, "src/index.js"),
  output: {
    path: path.join(__dirname, "dist"),
    filename: "bundle.js",
  },
  resolve: {
    extensions: [".js", ".jsx"],
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
      {
        test: /\.css$/i,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(jpe?g|png|gif|svg)$/i,
        loader: "file-loader",
      },
    ],
  },
  plugins: [new HtmlWebpackPlugin({ template: "./public/index.html" })],
  devServer: {
    hot: true, // hot reloading
    port: 3000, // port on which server will run
    open: true, // open browser automatically on start
    allowedHosts: ["http://127.0.0.1:3000", "http://localhost:3000"],
    proxy: [
      {
        context: ["/"],
        target: "http://127.0.0.1:8005",
        changeOrigin: true,
        secure: false,
      },
    ],
  },
};
