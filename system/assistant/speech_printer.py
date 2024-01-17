from cereal.messaging import SubMaster

sm = SubMaster(["speechToText"])

while True:
    sm.update(0)
    if sm.updated["speechToText"]:
        if sm["speechToText"].finalResultReady:
            print(f'Final result: {sm["speechToText"].result}')
        else:
            print(f'Interim result: {sm["speechToText"].result}')