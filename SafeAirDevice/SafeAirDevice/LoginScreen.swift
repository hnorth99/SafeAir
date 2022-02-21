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
            let humidity = snapshot.value as? Double ?? 0.005;
            print(humidity)
        });
    }
}
    /*
    @IBOutlet weak var tableView: UITableView!
    var
     tasks = [String]()
    var ref: DatabaseReference?
    var databaseHandle : DatabaseHandle?
    var postData = [String]()
    var handle: AuthStateDidChangeListenerHandle?
    
    private let label: UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.text = "Log In"
        label.font = .systemFont(ofSize: 24, weight: .semibold)
        return label
    } ()
    

    
    private let emailField: UITextField = {
        let emailField = UITextField()
        emailField.placeholder = "Email Address"
        emailField.layer.borderWidth = 1
        emailField.layer.borderColor = UIColor.black.cgColor
        emailField.autocapitalizationType = .none
        emailField.leftViewMode = .always
        emailField.leftView = UIView(frame:CGRect(x:0,
                                     y:0,
                                     width:5,
                                     height:0))
        return emailField
    } ()
    
    private let passwordField: UITextField = {
        let passField = UITextField()
        passField.placeholder = "Password"
        passField.layer.borderWidth = 1
        passField.isSecureTextEntry = true
        passField.layer.borderColor = UIColor.black.cgColor
        passField.leftViewMode = .always
        passField.leftView = UIView(frame:CGRect(x:0,
                                     y:0,
                                     width:5,
                                     height:0))
        return passField
    } ()
    
    private let button: UIButton = {
        let button = UIButton()
        button.backgroundColor = .systemGreen
        button.setTitleColor(.white, for:.normal)
        button.setTitle("Continue", for:.normal)
        return button
    } ()

    private let signOutButton: UIButton = {
        let button = UIButton()
        button.backgroundColor = .systemGreen
        button.setTitleColor(.white, for:.normal)
        button.setTitle("Log Out", for:.normal)
        return button
    } ()
    

    override func viewDidLoad() {
        super.viewDidLoad()
        view.addSubview(label)
        view.addSubview(emailField)
        view.addSubview(passwordField)
        view.addSubview(button)

        
        button.addTarget(self, action: #selector(didTapButton), for: .touchUpInside)
        if FirebaseAuth.Auth.auth().currentUser != nil {
            label.isHidden = true
            emailField.isHidden = true
            passwordField.isHidden = true
            button.isHidden = true
            
            view.addSubview(signOutButton)
            signOutButton.frame = CGRect(x:20,
                                         y:150,
                                         width: view.frame.size.width-40,
                                         height:52)
            signOutButton.addTarget(self, action: #selector(logOutTapped), for: .touchUpInside)

            
        }
    }

    @objc private func logOutTapped() {
        do {
            try FirebaseAuth.Auth.auth().signOut()
            label.isHidden = false
            emailField.isHidden = false
            passwordField.isHidden = false
            button.isHidden = false
            signOutButton.removeFromSuperview()
        }
        catch {
            print("An error occured")
            
        }
    }
    

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        label.frame = CGRect(x:0, y: 100, width: view.frame.size.width, height:80)
        
        emailField.frame = CGRect(x:20, y: label.frame.origin.y + label.frame.size.height+10,
                                  width: view.frame.size.width-40, height:80)
        
        passwordField.frame = CGRect(x:20,
                                     y: emailField.frame.origin.y+emailField.frame.size.height+10,
                                     width: view.frame.size.width-40,
                                     height:50)
        
        button.frame = CGRect(x:20,
                              y: passwordField.frame.origin.y+passwordField.frame.size.height+30,
                              width: view.frame.size.width-40,
                              height:52)

    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        if FirebaseAuth.Auth.auth().currentUser == nil {
            emailField.becomeFirstResponder()

        }
        
    }
    @objc private func didTapButton() {
        print("Continue button tapped")
        guard let email = emailField.text, !email.isEmpty,
              let password = passwordField.text, !password.isEmpty else {
                  print("Missing field data")
                  return
              }
        FirebaseAuth.Auth.auth().signIn(withEmail: email, password: password, completion: {[weak self] result, error in
            guard let strongSelf = self else {
                return
            }
            guard error == nil else {
                strongSelf.showCreateAccount(email: email, password: password)
                return
            }
            print ("You have signed in")
            strongSelf.label.isHidden=true
            strongSelf.emailField.isHidden=true
            strongSelf.passwordField.isHidden=true
            strongSelf.button.isHidden=true
            strongSelf.emailField.resignFirstResponder()
            strongSelf.passwordField.resignFirstResponder()

        })
        
    }

    override func viewWillDisappear(_ animated: Bool) {
      super.viewWillDisappear(animated)
      navigationController?.setNavigationBarHidden(false, animated: false)
    }
    
    func showCreateAccount(email: String, password: String) {
        let alert = UIAlertController(title: "Create Account",
                                      message: "Would you like to create an account",
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Continue",
                                      style: .default,
                                      handler: {_ in
                                    
            FirebaseAuth.Auth.auth().createUser(withEmail: email, password: password, completion: { [weak self]result, error in
                guard let strongSelf = self else {
                    return
                }
                guard error == nil else {
                    print("Account creation failed")
                    return
                }
                print ("You have signed in")
                strongSelf.label.isHidden=true
                strongSelf.emailField.isHidden=true
                strongSelf.passwordField.isHidden=true
                strongSelf.button.isHidden=true
                strongSelf.emailField.resignFirstResponder()
                strongSelf.passwordField.resignFirstResponder()
            })
            
        }))
        alert.addAction(UIAlertAction(title: "Cancel",
                                      style: .cancel,
                                      handler: {_ in
            
        }))
        
        present(alert, animated: true)
    }

*/

    /*
extension ViewController: UITableViewDelegate {

    func tableView(_tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
    }

}

extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tasks.count
    }
    
    func tableView(_ tableView:UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        cell.textLabel?.text = tasks[indexPath.row]
        return cell
    }
    
}
     */
