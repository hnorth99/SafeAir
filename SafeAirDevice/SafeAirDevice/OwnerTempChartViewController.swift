//  OwnerTempChartViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 3/4/22.

// The following are packages imported for this program.
import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase

// The OwnerTempChartViewController is designed to display the historic metrics of the temperature measurements. These measurements are fetched from the firebase real-time database and then displayed as a line chart. In addition
// the owner has the able to see customer complaints as well as edit the constraints that serve as predefined setpoints to regulating external devices.
class OwnerTempChartViewController: UIViewController, ChartViewDelegate {
    // The following code initializes UILabels and TextFields for this
    // program.
    @IBOutlet weak var currentTempConst: UILabel!
    @IBOutlet weak var editMaxTemp: UITextField!
    @IBOutlet weak var TempComp: UILabel!
    
    // The following code initializes a variable appDelegate.
    let appDelegate = UIApplication.shared.delegate as! AppDelegate
    var lineChart = LineChartView()

    override func viewDidLoad() {
        super.viewDidLoad()
        lineChart.delegate = self
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        // The following line configures the display of the graph.
        lineChart.frame = CGRect(x:0, y:0, width: self.view.frame.size.width,
                                 height: self.view.frame.size.width)
        lineChart.center = view.center
        view.addSubview(lineChart)
        // The varaible test is created of type 'ChartDataEntry' that will hold the
        // values that will then be plotted.
        var test = [ChartDataEntry]()
        //var entries = [Double]()
        // The following appDelegate function indexes into the node 'HunterApt to get
        // access to all the associated children.
        appDelegate.ref.child("HunterApt").getData(completion: { error, snapshot in
            guard error == nil else {
                print(error!.localizedDescription)
                return;
            }
            // 'Temperature' and 'Time' are then the selected child node of Hunter Apt.
            // These values will serve as the x and y points for the graph configured.
            let entries = snapshot.value as? NSDictionary
            let temp = entries?["temperature"] as? [Double] ?? [0.0005]
            //var time = entries?["time"] as? [NSString] ?? [""]
            for x in (temp.count-70)..<temp.count {
                test.append(ChartDataEntry(x: Double(x-70), y: Double(temp[x])))
            }
            // The following displays the current temperature complaints.
            let comp = entries?["temperaturecomplaints"] as? Double ?? 0.0005
            self.TempComp.text = String(comp)
            //The folllwing displays the current temp constraint.
            let tempCons = entries?["temp_min"] as? String ?? ""
            self.currentTempConst.text = String(tempCons) + " °C"

            
            // The following code creates modifications to the graph being displayed.
            // For example eliminating the xAxis labels and only having the Y axis labels
            // appear on one side of the graph instead of both.
            self.lineChart.xAxis.drawLabelsEnabled = false
            self.lineChart.rightAxis.drawLabelsEnabled = false
            self.lineChart.legend.enabled = false
            
            // The following line plots the data in test onto the graph.
            let set = LineChartDataSet(entries: test)
            // The following color modifications are made to the graph.
            set.colors = ChartColorTemplates.material()
            let data = LineChartData(dataSet: set)
            self.lineChart.data = data
        }
        )
    }
    // The following function apply saves the newly updated temperature min constraint to the database.
    @IBAction func apply_pressed(_ sender: Any) {
        appDelegate.ref.child("HunterApt").child("temp_min").setValue(editMaxTemp.text);
    }
}
