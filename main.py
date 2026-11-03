import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        app_dir="backend",
        host="0.0.0.0",
        port=5000,
    )
