//  LocationMetrics.swift
//  SafeAirDevice
//  Created by Rafka Daou on 2/24/22.

// The following are packages imported for this program.
import Foundation
import UIKit
import FirebaseAuth
import FirebaseDatabase

// The following class is designed to fetch data from the firebase realtime
// database and display the realtime metrics through the app.
class LocationMetrics: UIViewController {
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    
    // The following code initiatizes labels.
    @IBOutlet weak var locationLabel: UILabel!
    @IBOutlet weak var capacityLabel: UILabel!
    @IBOutlet weak var tempLabel: UILabel!
    @IBOutlet weak var humidityLabel: UILabel!
    @IBOutlet weak var co2Label: UILabel!
    
    // The following function viewDidLoad displays the location metrics location
    // as well as calls the function readData() to periodically update the real
    // time measurements being displayed.
    override func viewDidLoad() {
        super.viewDidLoad()
        locationLabel.text = appDelegate.location
        readData()
    }
    
    // The function readData is designed to fetch real time measurements of the
    // temperature, humidity, carbon dioxide and capacity levels in a room. This infromation
    // will then be displayed in the app.
    func readData() {
        appDelegate.ref.child(appDelegate.location).observe(DataEventType.value, with: {
                (snapshot) in
            // The following converts the data receieved as a snapshot to an NSDicitionary.
            let dict = snapshot.value as? NSDictionary
            let capacity = dict?["currentcapacity"] as? Double ?? 0.0
            //let temperature = dict?["temperature"] as? Double ?? 0.0
            let temperature = dict?["temperature"] as? [Double] ?? [0.0]
            //let humidity = dict?["humidity"] as? Double ?? 0.0
            let humidity = dict?["humidity"] as? [Double] ?? [0.0]
            //let co2 = dict?["co2"] as? Double ?? 0.0
            let co2 = dict?["co2"] as? [Double] ?? [0.0]
            
            let max_cap = dict?["maxcapacity"] as? String ?? ""
            let min_temp = dict?["temp_min"] as? String ?? ""
            let max_humid = dict?["humidity_max"] as? String ?? ""
            let min_humid = dict?["humidity_min"] as? String ?? ""
            let max_co2 = dict?["co2_max"] as? String ?? ""
            // The following lines assign each label with its respective value.
            self.capacityLabel.text = String(capacity)
            self.tempLabel.text = String(temperature[temperature.count - 1]) + " °C"
            self.humidityLabel.text = String(humidity[humidity.count - 1]) + " %"
            self.co2Label.text = String(co2[co2.count - 1]) + " ppm"
            
            // This program is also designed to regulate the air condition within the room
            // by monitoring when the air quality exceeds predetermined set points. The following
            // if-else statements notify anyone viewing the data if the measurements are safe
            // or not by labeling them with green or red.
            if (capacity > Double(max_cap) ?? 0.0) {
                self.capacityLabel.textColor = UIColor.red
            } else {
                self.capacityLabel.textColor = UIColor.green
            }
            if (temperature[temperature.count - 1] < Double(min_temp) ?? 0.0) {
                self.tempLabel.textColor = UIColor.red
             } else {
                self.tempLabel.textColor = UIColor.green
             }
            if (humidity[humidity.count - 1] < Double(min_humid) ?? 0.0 || humidity[humidity.count - 1] > Double(max_humid) ?? 0.0) {
                self.humidityLabel.textColor = UIColor.red
            } else {
                self.humidityLabel.textColor = UIColor.green
            }
            if (co2[co2.count - 1] > Double(max_co2) ?? 0.0) {
                self.co2Label.textColor = UIColor.red
            } else {
                self.co2Label.textColor = UIColor.green
            }
        }
        )
    }
}
