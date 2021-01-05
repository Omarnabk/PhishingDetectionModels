from main.model.services import ServiceRoutines

if __name__ == "__main__":
    fnc_service = ServiceRoutines()

    CSA_FN = '219.zenni-littlekids-boy-girls-md.jpg'
    print(fnc_service.classify_file_name_charngramTFIDF(text=CSA_FN))

    FN = 'AdobeGCLaunchEvent.db'
    print(fnc_service.classify_file_name_charngramTFIDF(text=FN))
