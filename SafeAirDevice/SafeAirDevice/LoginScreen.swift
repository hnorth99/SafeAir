//  ViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 2/18/22.

// The following are packages imported for this program.
import UIKit
import FirebaseAuth
import FirebaseDatabase

// The class LoginScreen is designed to greet an individual upon opening the app.
// The purpose of this class is to setup a view for either a customer or an owner
// to login and access metrics with the repsect to the specified location.

class LoginScreen: UIViewController{
    
    // The following code initializes a variable appDelegate.
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    
    // The following code intiializes cust_locid, owner_locid and onwer_pass as a text field.
    // This fields are used for the individual to enter in their information.
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
        // The following code checks to see if there is retrievable data at that location,
        // returning an error otherwise.
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
    // The folloiwng function is designed to check if the location and password the owner
    // enters correctly match what is in the system.
    @IBAction func owner_login_pressed(_ sender: Any) {
        // Sets default value of owner_loc_id and owner_password to be an empty string,
        // otherwise will be what the owner enters as the location and password.
        let owner_loc_id = owner_locid.text ?? "empty"
        let owner_password = owner_pass.text ?? "empty"
        
        self.appDelegate.ref.child(owner_loc_id).child("password").observe(DataEventType.value, with: { snapshot in
        // Response holds a string of the location the owner enters.
        // Response holds a string of the password the owner enters.
        let response = snapshot.exists();
        if (!response) {
            // If the location does not exist in the system- the owner is alerted.
            let alertController = UIAlertController(title: "Location ID Does Not Exist", message: "Please retype correct ID", preferredStyle: .alert)
            alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in alertController.dismiss(animated: true, completion: nil)
            })
            )
            self.present(alertController, animated: true, completion: nil)
          } else if(owner_password != (snapshot.value as? String ?? "")) {
              // If the password does not match what is in the system- the owner is alerted.
              let alertController = UIAlertController(title: "Incorrect Password", message: "Please retype password", preferredStyle: .alert)
                 alertController.addAction(UIAlertAction(title: "Okay", style: UIAlertAction.Style.default, handler: {(action) in alertController.dismiss(animated: true, completion: nil)
                 })
                 )
                 self.present(alertController, animated: true, completion: nil)
          }
            // If the owner has successfully entered the correct information for a given location,
            // they are then presented with the air quality metrics for the specified location.
            else {
             self.appDelegate.location = owner_loc_id
             self.appDelegate.password = owner_password
             self.performSegue(withIdentifier: "OwnerToStatView", sender: self)
     }
     });
 }
}
