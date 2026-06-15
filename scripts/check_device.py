from openvino import Core


def main() -> None:
    core = Core()
    print(core.available_devices)


if __name__ == "__main__":
    main()