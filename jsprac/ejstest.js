
///////////////////////////////////////////////////////
// Imports

var ejs = require('ejs')
const { Console } = require('console');

///////////////////////////////////////////////////////
// Node Templates

/*
    Basic Template Syntax:
    (TODO - This is very incorrect/incomplete. Needs to allow for multiple out/inputs, options, etc.)
        - outputVar: name of variable created from this node's operation
        - inputVar: same as above for input
 */

/**
 * Template for creating a spark df object
 * @type {string}
 *
 * TODO - update to support multiple formats?
 */
let dataTemplate = '<%= outputVar1 %> = <%=sparkVar%>.read.load("<%=inputVar1%>")';

let linRegTemplate = 'lr = LinearRegression(maxIter=<%=inputVar1%>, regParam=<%=inputVar2%>, elasticNetParam=<%=inputVar3%>)' +
    '\n<%=outputVar1%> = lr.fit(<%=inputVar4%>)';

let printToConsoleTemplate = 'print(<%=inputVar1%>.summary)';


///////////////////////////////////////////////////////
// NodeData JSON structures
// Note: these will be generated by WebGME plugin, and that
//          the exact syntax might be off (for children/parents, etc.)

let dataNode = {
    inputFile: "example.json",
    guid: "123456data",
    requirements: []
};

let linRegNode = {
    guid: "123456LinReg",
    maxIter: 10,
    regParam: 0.3,
    elasticNetParam: 0.8,
    inputDataNode: "123456data",  // TODO | need to find this programmatically
    requirements: [
        "from pyspark.ml.regression import LinearRegression"
    ]
};

let printToConsoleNode = {
    inputModel: "123456LinReg",
    guid: "123456print",
    requirements: []
};

let basic_nodes = [dataNode, linRegNode, printToConsoleNode];  // Not used in simplified code, but used in real impl

/////////////////////////////////////////////////////
// Convert node data to templatetized data format for ejs templates above
// TODO | Discover template type?

let sparkVar = 'spark';  // TODO | always start with a spark session being created. Need to have this be a node

let dataInput = {
    outputVar1: dataNode.guid,  // TODO | what if multiple 'outputs'?
    sparkVar: sparkVar,
    inputVar1: dataNode.inputFile
};

let linRegInput = {
    outputVar1: linRegNode.guid,
    inputVar1: linRegNode.maxIter,
    inputVar2: linRegNode.regParam,
    inputVar3: linRegNode.elasticNetParam,
    inputVar4: linRegNode.inputDataNode
};

let printToConsoleInput = {
    inputVar1: printToConsoleNode.inputModel
};

let nodes = [
    {node: dataNode, templateInput: dataInput, template: dataTemplate},
    {node: linRegNode, templateInput: linRegInput, template: linRegTemplate},
    {node: printToConsoleNode, templateInput: printToConsoleInput, template: printToConsoleTemplate}
];

////////////////////////////////////////////////////
// Execution

// First gather and add all imports to the file

function getImports(nodes){

    let importStatements = "";


    // TODO | import sparkSession each time?
    importStatements += "from pyspark.sql import SparkSession \n";

    for (node of nodes){
        for(req of node.node.requirements){
            importStatements += req + " \n";
        }
    }

    importStatements += "\n";
    return importStatements;
}

function createSparkSession(sparkSessName, appname){
    return sparkSessName + " = SparkSession.builder.appName(\"" + appname + "\").getOrCreate() \n \n"
}

/**
 * Nodes must be ordered sequentially
 * @param nodes
 */
function applyTemplates(nodes){

    let executionStatements = "";

    for (node of nodes){
        executionStatements += ejs.render(node.template, node.templateInput) + "\n \n"
    }

    return executionStatements + "\n"

}


// create output file
let outCode = "";

// add needed imports
outCode += getImports(nodes);

// create the spark session
outCode += createSparkSession(sparkVar, "app_name");

// add in main code
outCode += applyTemplates(nodes);


console.log(outCode);




// console.log(ejs.render(dataTemplate, data));



// Test code
// console.log('testing');
//
//
// let templ = "this is the title: <%= title %>";
//
// let result = ejs.render(templ, {title:'example_'});
//
// console.log(result);


