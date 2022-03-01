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
            let capacity = dict?["capacity"] as? Double ?? 0.0
            let temperature = dict?["temperature"] as? Double ?? 0.0
            let humidity = dict?["humidity"] as? Double ?? 0.0
            let co2 = dict?["co2"] as? Double ?? 0.0
            
            self.capacityLabel.text = String(capacity)
            self.tempLabel.text = String(temperature)
            self.humidityLabel.text = String(humidity)
            self.co2Label.text = String(co2)
            
            if (capacity > 1) {
                self.capacityLabel.textColor = UIColor.red
            } else {
                self.capacityLabel.textColor = UIColor.green
            }
            if (temperature < 11) {
                self.tempLabel.textColor = UIColor.red
            } else {
                self.tempLabel.textColor = UIColor.green
            }
            if (humidity < 40 || humidity > 60) {
                self.humidityLabel.textColor = UIColor.red
            } else {
                self.humidityLabel.textColor = UIColor.green
            }
            if (co2 > 1500) {
                self.co2Label.textColor = UIColor.red
            } else {
                self.co2Label.textColor = UIColor.green
            }
        }
        )
        
    }
    
}
