//
//  LocationMetrics.swift
//  SafeAirDevice
//
//  Created by Rafka Daou on 2/24/22.
//

import Foundation
import UIKit
import FirebaseAuth
import FirebaseDatabase

class LocationMetrics: UIViewController {
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    
    @IBOutlet weak var locationLabel: UILabel!

    @IBOutlet weak var capacityLabel: UILabel!
    @IBOutlet weak var tempLabel: UILabel!
    @IBOutlet weak var humidityLabel: UILabel!
    @IBOutlet weak var co2Label: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        locationLabel.text = appDelegate.location
        readData()
        
        // Do any additional setup after loading the view.
    }
    
    func readData() {
        appDelegate.ref.child(appDelegate.location).observe(DataEventType.value, with: {
                (snapshot) in
            let dict = snapshot.value as? NSDictionary
            self.capacityLabel.text = String(dict?["capacity"] as? Double ?? 0.0)
            self.tempLabel.text = String(dict?["temperature"] as? Double ?? 0.0)
            self.humidityLabel.text = String(dict?["humidity"] as? Double ?? 0.0)
            self.co2Label.text = String(dict?["co2"] as? Double ?? 0.0)
        }
        )
        
    }
    
}
