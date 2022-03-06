//
//  ViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 2/18/22.

import UIKit
import FirebaseAuth
import FirebaseDatabase

class LoginScreen: UIViewController{
    // The following code initializes a variable appDelegate.
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    // The following code intiializes cust_locid as a text field. This is used
    // to indicate what location we are expecting to receive air quality metrics for.
    @IBOutlet weak var cust_locid: UITextField!
    @IBOutlet weak var owner_locid: UITextField!
    @IBOutlet weak var owner_pass: UITextField!
    
    // The following fuction cust_login_pressed is designed to create the login screen
    // function for the application. When the user opens the app they are greeted with
    // a login screen that requires them to enter in the location ID they are requesting
    // to get air quality metrics for. If the user enters a location that is not rendered
    // by the system they will be alerted.
    @IBAction func cust_login_pressed(_ sender: Any) {
        let ref = appDelegate.ref
        // Sets default value of loc_id to be an empty string, otherwise will be what
        // the user enters as the location.
        let loc_id = cust_locid.text ?? "empty"
        //
        ref?.child(loc_id + "/humidity").getData(completion:  { error, snapshot in
        guard error == nil else {
            print(error!.localizedDescription)
            return;
        }
        // Response holds a string of the location the user enters.
        let response = snapshot.exists();
         if (!response) {
             // If the location does not exist in the system- the user is alerted.
             let alertController = UIAlertController(title: "Location ID Does Not Exist", message: "Please retype correct ID", preferredStyle: .alert)
            alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in alertController.dismiss(animated: true, completion: nil)
            })
            )
            self.present(alertController, animated: true, completion: nil)
        }
            // If the location ID is successfully entered the app segues into a new page
            else {
            self.appDelegate.location = loc_id
            self.performSegue(withIdentifier: "CustomerLoginToCustomerStat", sender: self)
        }
        });
    }
    
    
    @IBAction func owner_login_pressed(_ sender: Any) {
        // Sets default value of loc_id to be an empty string, otherwise will be what
        // the user enters as the location.
        let owner_loc_id = owner_locid.text ?? "empty"
        let owner_password = owner_pass.text ?? "empty"

        self.appDelegate.ref.child(owner_loc_id).child("password").observe(DataEventType.value, with: { snapshot in
        // Response holds a string of the location the user enters.
        // Response holds a string of the location the user enters.
        let response = snapshot.exists();
        if (!response) {
            // If the location does not exist in the system- the user is alerted.
            let alertController = UIAlertController(title: "Location ID Does Not Exist", message: "Please retype correct ID", preferredStyle: .alert)
            alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in alertController.dismiss(animated: true, completion: nil)
            })
            )
            self.present(alertController, animated: true, completion: nil)
          } else if(owner_password != (snapshot.value as? String ?? "")) {
              // If the password does not match what is in the system- the user is alerted.
              let alertController = UIAlertController(title: "Incorrect Password", message: "Please retype password", preferredStyle: .alert)
                 alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in alertController.dismiss(animated: true, completion: nil)
                 })
                 )
                 self.present(alertController, animated: true, completion: nil)
          } else {
             self.appDelegate.location = owner_loc_id
             self.appDelegate.password = owner_password
             self.performSegue(withIdentifier: "OwnerToStatView", sender: self)
     }
     });
 }
}
