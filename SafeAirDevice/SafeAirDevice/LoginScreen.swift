//
//  ViewController.swift
//  SafeAirDevice
//
//  Created by Rafka Daou on 2/18/22.
//

import UIKit
import FirebaseAuth
import FirebaseDatabase

class LoginScreen: UIViewController{
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    
    @IBOutlet weak var cust_locid: UITextField!
    
    @IBAction func cust_login_pressed(_ sender: Any) {
        let ref = appDelegate.ref
        let loc_id = cust_locid.text ?? "empty"
        
        ref?.child(loc_id + "/humidity").getData(completion:  { error, snapshot in
          guard error == nil else {
            print(error!.localizedDescription)
            return;
          }
            let response = snapshot.exists();
            print(response)
        
            if (!response) {
                print("here")
                let alertController = UIAlertController(title: "Location ID Does Not Exist", message: "Please retype correct ID", preferredStyle: .alert)
                alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in
                        alertController.dismiss(animated: true, completion: nil)
        
                    })
                )
                self.present(alertController, animated: true, completion: nil)
            } else {
                self.appDelegate.location = loc_id
                self.performSegue(withIdentifier: "CustomerLoginToCustomerStat", sender: self)
            }
        });
    }
    
}

