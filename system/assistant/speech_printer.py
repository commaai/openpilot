from cereal.messaging import SubMaster

sm = SubMaster(["speechToText"])


def main():
    while True:
        navigate = False
        result: str = ""
        sm.update(0)
        if sm.updated["speechToText"]:
            result = sm["speechToText"].result
            if not sm["speechToText"].finalResultReady:
                print(f'Interim result: {result}')
            else:
                print(f'Final result: {result}')
                if "navigate" in result or "directions" in result:
                    navigate = True

        if navigate:
            address = result.split("to ")[1]
            print(f'Getting Directions to {address}')
            
if __name__ == "__main__":
    main()

