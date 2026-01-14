import uvicorn

from app.web.server import create_app


def main():
    app = create_app(web_dir="web")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

