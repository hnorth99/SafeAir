//  OwnerCapChartViewController.swift
//  SafeAirDevice
//  Created by Rafka Daou on 3/4/22.

// The following are packages imported for this program
import UIKit
import UIKit
import Charts
import TinyConstraints
import Foundation
import FirebaseAuth
import FirebaseDatabase

// The OwnerCapChartViewController is designed to display the historic metrics of the capacity measurements. These measurements are fetched from the firebase real-time database and then displayed as a line chart. In addition
// the owner has the able to see customer complaints as well as edit the constraints that serve as predefined setpoints to regulating external devices.
class OwnerCapChartViewController: UIViewController, ChartViewDelegate  {
   
    // The folloiwng code initializes capComp and
    // currentCapCons as a UILabel.
    @IBOutlet weak var capComp: UILabel!
    // The following code initializes editCapacity as a text field.
    @IBOutlet weak var editCapacity: UITextField!
    @IBOutlet weak var currentCapCons: UILabel!
    
    // The folloiwng code initiaizes a variable appDelegate.
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
            // 'currentcpacity' and 'Time' are then the selected child node of Hunter Apt.
            // These values will serve as the x and y points for the graph configured.
            let entries = snapshot.value as? NSDictionary
            let cap = entries?["currentcapacity"] as? Double ?? 0.0005
            test.append(ChartDataEntry(x: Double(1), y: Double(cap)))

            // The following displays the current capacitycomplaints.
            let comp = entries?["capacitycomplaints"] as? Double ?? 0.0005
            self.capComp.text = String(comp)
            
            // The following displays the current capcaity constraint.
            let const = entries?["maxcapacity"] as? String ?? ""
            self.currentCapCons.text = String(const)
           
            
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
    // The following function apply saves the newly updated capacity constraint to the database.
    @IBAction func apply(_ sender: Any) {
        appDelegate.ref.child("HunterApt").child("maxcapacity").setValue(editCapacity.text);
    }
}
