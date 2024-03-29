/*globals define*/
/*jshint node:true, browser:true*/

/**
 * This plugin Generates python Spark code from a well defined model.
 * Must be run server side, as it uses the ejs node library.
 */

const ejs = require('ejs');

define([
    'plugin/PluginConfig',
    'text!./metadata.json',
    'plugin/PluginBase'
], function (
    PluginConfig,
    pluginMetadata,
    PluginBase) {
    'use strict';

    /////////////////////////////////////////////////////////////////////////////
    // Tree definition and methods.

    /**
     * Tree Node Object
     * @param data
     * @constructor
     */
    function Node(data) {
        this.data = data;
        this.parent = null;
        this.children = [];
    }

    /**
     * Tree Object
     * @param data
     * @constructor
     */
    function Tree(data) {
        let node = new Node(data);
        this._root = node;
    }

    /**
     * Traverses the tree in a depth first order
     * @param callback
     */
    Tree.prototype.traverseDF = function(callback) {

        // this is a recurse and immediately-invoking function
        (function recurse(currentNode) {
            // step 2
            for (let i = 0, length = currentNode.children.length; i < length; i++) {
                // step 3
                recurse(currentNode.children[i]);
            }

            // step 4
            callback(currentNode);

            // step 1
        })(this._root);

    };

    /**
     * Traverses the tree breath first
     * @param callback
     */
    Tree.prototype.traverseBF = function(callback) {
        let queue = [];

        queue.push(this._root);

        let currentTree = queue.shift();

        while(currentTree){
            for (let i = 0, length = currentTree.children.length; i < length; i++) {
                queue.push(currentTree.children[i]);
            }

            callback(currentTree);
            currentTree = queue.shift();
        }
    };

    /**
     * Applies the callback to each node in 'traversal' order
     * @param callback
     * @param traversal
     */
    Tree.prototype.contains = function(callback, traversal) {
        traversal.call(this, callback);
    };

    /**
     * Adds the data to the tree as a child of 'toData'
     * @param data
     * @param toData
     * @param traversal
     */
    Tree.prototype.add = function(data, toData, traversal) {
        let child = new Node(data),
            parent = null,
            callback = function(node) {
                if (node.data === toData) {
                    parent = node;
                }
            };

        this.contains(callback, traversal);

        if (parent) {
            parent.children.push(child);
            child.parent = parent;
        } else {
            throw new Error('Cannot add node to a non-existent parent.');
        }
    };

    /**
     * Removes the data from the tree
     * @param data
     * @param fromData
     * @param traversal
     * @returns {*}
     */
    Tree.prototype.remove = function(data, fromData, traversal) {
        let tree = this,
            parent = null,
            childToRemove = null,
            index;

        let callback = function(node) {
            if (node.data === fromData) {
                parent = node;
            }
        };

        this.contains(callback, traversal);

        if (parent) {
            index = findIndex(parent.children, data);

            if (index === undefined) {
                throw new Error('Node to remove does not exist.');
            } else {
                childToRemove = parent.children.splice(index, 1);
            }
        } else {
            throw new Error('Parent does not exist.');
        }

        return childToRemove;
    };

    function findIndex(arr, data) {
        let index;

        for (let i = 0; i < arr.length; i++) {
            if (arr[i].data === data) {
                index = i;
            }
        }

        return index;
    }

    pluginMetadata = JSON.parse(pluginMetadata);

    /**
     * Initializes a new instance of SparkCodeGen.
     * @class
     * @augments {PluginBase}
     * @classdesc This class represents the plugin SparkCodeGen.
     * @constructor
     */
    var SparkCodeGen = function () {
        // Call base class' constructor.
        PluginBase.call(this);
        this.pluginMetadata = pluginMetadata;
    };

    /**
     * Metadata associated with the plugin. Contains id, name, version, description, icon, configStructue etc.
     * This is also available at the instance at this.pluginMetadata.
     * @type {object}
     */
    SparkCodeGen.metadata = pluginMetadata;

    // Prototypical inheritance from PluginBase.
    SparkCodeGen.prototype = Object.create(PluginBase.prototype);
    SparkCodeGen.prototype.constructor = SparkCodeGen;

    /**
     * Main function for the plugin to execute. This will perform the execution.
     * Notes:
     * - Always log with the provided logger.[error,warning,info,debug].
     * - Do NOT put any user interaction logic UI, etc. inside this method.
     * - callback always has to be called even if error happened.
     *
     * @param {function(string, plugin.PluginResult)} callback - the result callback
     */
    SparkCodeGen.prototype.main = function (callback) {

        var self = this;
        var activeNode = this.activeNode;
        var core = this.core;
        var logger = this.logger;

        var jsonTree = {
            // Root of node jsonTree
            name: 'ROOT'
        };

        // An array of the meta nodes
        var metaArray = [];

        // The python script that is created as an artifact
        var artifact = null;

        // Set of all nodes present in current context
        var allNodes = new Set();

        // A dictionary of the data from each component in the supplied model
        var componentDataDictionary = {};

        // The statements to import in the python code
        var imports = "from pyspark.sql import SparkSession\n";


        /**
         * Returns the string generated by applying the supplied EJS template (stored on the
         * meta node) using the node's data
         * @param componentData
         * @returns {*}
         */
        function applyTemplate(componentData){
            return ejs.render(String(componentData.EJSTemplate), componentData);
        }

        /**
         * Return a set of all the names of Component nodes in current context
         * @param nodes: all nodes in current context of any type
         * @returns {Set}
         */
        function getAllComponents(nodes, nodeMap){
            let allComponents = new Set();

            // Loop through all nodes
            for(let node of nodes){
                let metaTypeNode = core.getMetaType(node);
                let metaType = core.getAttribute(metaTypeNode, 'name');

                if (metaType !== "Model" &&
                    metaType !== "Connection" &&
                    metaType !== "Input" &&
                    metaType !== "Output" &&
                    metaType !== "Port"){


                    // Retrieve the imports for this node
                    if(String(core.getAttribute(node, 'Imports')) !== '')
                        imports += String(core.getAttribute(node, 'Imports')) + "\n";

                    allComponents.add(node);
                    componentDataDictionary[core.getGuid(node)] = {
                        data: getComponentJsonData(node, nodeMap),
                        connections: {src: [], dst: []}
                    };
                }
            }
            // Add all connections
            for(let node of nodes){
                let metaTypeNode = core.getMetaType(node);
                let metaType = core.getAttribute(metaTypeNode, 'name');

                if (metaType === "Connection"){

                    let srcPath = core.getPointerPath(node, 'src');
                    let dstPath = core.getPointerPath(node, 'dst');

                    if (srcPath && dstPath) {

                        let connGuid = 'a' + String(core.getGuid(node)).replace(/-/g, "_");

                        let srcPortNode = nodeMap[srcPath];
                        let dstPortNode = nodeMap[dstPath];

                        let srcPortName = core.getAttribute(srcPortNode, 'name');
                        let dstPortName = core.getAttribute(dstPortNode, 'name');

                        let srcComp = core.getParent(srcPortNode);
                        let dstComp = core.getParent(dstPortNode);

                        componentDataDictionary[core.getGuid(srcComp)].connections.src.push({
                            guid: connGuid,
                            srcPort: srcPortName,
                            srcNode: srcComp,
                            destPort: dstComp,
                            destNode: dstComp
                        });

                        componentDataDictionary[core.getGuid(dstComp)].connections.dst.push({
                            guid: connGuid,
                            srcPort: srcPortName,
                            srcNode: srcComp,
                            destPort: dstComp,
                            destNode: dstComp
                        });

                        componentDataDictionary[core.getGuid(srcComp)].data[srcPortName] = connGuid;
                        componentDataDictionary[core.getGuid(dstComp)].data[dstPortName] = connGuid;
                    }
                }
            }

            logger.info('component data ', componentDataDictionary);

            return allComponents;
        }

        /**
         * Return a sequence graph (i.e. a tree of the components) representing the order in which they should
         * be created.
         * @param components
         * @param dataDictionary
         * @returns {Tree}
         */
        function getSequenceGraph(components, dataDictionary) {

            // create a set of the nodes not in the sequence
            let compNotInSeq = new Set();
            for (let node of components){
                compNotInSeq.add(core.getGuid(node));
            }

            // create tree
            let root = '';
            let componentSequence = new Tree(root);

            let currLevel = [];
            let nodesInTree = new Set();
            let delayNodes = new Set();

            // get source nodes
            for (let node of components){
                let srcs = dataDictionary[core.getGuid(node)].connections.src;
                let dsts = dataDictionary[core.getGuid(node)].connections.dst;
                if(srcs.length > 0 && dsts.length === 0) {
                    currLevel.push(node);
                    nodesInTree.add(node);
                    componentSequence.add(core.getGuid(node),
                        root, componentSequence.traverseBF);

                    compNotInSeq.delete(core.getGuid(node));
                }
            }

            while(compNotInSeq.size > 0){
                let newChildren = new Set();

                for(let node of currLevel){
                    let srcs = dataDictionary[core.getGuid(node)].connections.src;

                    for (let child of srcs){
                        newChildren.add(child.destNode);
                    }
                }

                // add the delayed nodes:

                let twoSets = [newChildren, delayNodes];

                let validatedNodes = new Set();
                let newDelayNodes = new Set();

                for (let set of twoSets){
                    for(let node of set){

                        let isValid = true;
                        let parents = [];
                        let destConn = dataDictionary[core.getGuid(node)].connections.dst

                        for (let conn of destConn){
                            parents.push(conn.srcNode);
                        }

                        // check that all parents are in the tree
                        for (let parent of parents){
                            if (!nodesInTree.has(parent)){
                                isValid = false;
                            }
                        }

                        // If parents are in the tree, add to sequence. Else delay the node
                        if (isValid){
                            validatedNodes.add(node);
                        }else {
                            newDelayNodes.add(node);
                        }
                    }
                }

                let newCurLevel = [];

                for(let node of validatedNodes){
                    nodesInTree.add(node);
                    newCurLevel.push(node);
                    componentSequence.add(core.getGuid(node),
                        core.getGuid(currLevel[0]),
                        componentSequence.traverseBF);
                    compNotInSeq.delete(core.getGuid(node));
                }
                delayNodes = newDelayNodes;
                currLevel = newCurLevel;
            }

            return componentSequence;

        }

        /**
         * Generates spark code implementation given a sequence tree of nodes
         * @param sequence
         * @param dictionary
         * @returns {string}
         */
        function createSparkCodeFromSequence(sequence, dictionary){

            let stringOutput = "";

            stringOutput += imports + "\n";

            let appname = "generatedApplicaton";

            stringOutput += "spark = SparkSession.builder.appName(\"" + appname + "\").getOrCreate() \n"

            // iterate through the sequence level by level, generating the spark implementation
            sequence.traverseBF(node => {
                logger.info('node.data', node.data);
                if(node.data !== ''){
                    stringOutput += applyTemplate(dictionary[node.data].data) + "\n";
                }

            });

            return stringOutput;

        }
        // Check if the node is a meta node
        function isMeta(node) {

            let name = core.getAttribute(node, 'name');
            if (self.META.hasOwnProperty(name) && self.core.getPath(self.META[name]) === self.core.getPath(node)) {
                return true;
            } else {
                return false;
            }
        }

        // Get node's meta type
        function getMetaType(node) {

            let metaNode = null;
            if (isMeta(node)) {
                metaNode = node;
            } else {
                metaNode = self.core.getBase(node);
            }

            return metaNode;

        };

        // add a meta node to the meta array
        function addMetaNode(node) {
            let metaNodeData = {};

            // Get name
            metaNodeData.name = core.getAttribute(node, 'name')

            // Get the full path of the meta node
            metaNodeData.path = core.getPath(node)

            // Get the number of children
            metaNodeData.nbrOfChildren = core.getChildrenRelids(node).length;

            // Get the base node of the META node
            let baseNode = core.getBase(node)
            if (baseNode !== null) {
                metaNodeData.base = core.getAttribute(baseNode, 'name');
            } else {
                metaNodeData.base = null;
            }

            // push the the json to the state array
            metaArray.push(metaNodeData)
        }

        /**
         * add all nodes in the model to the set allNodes
         * @param node
         * @param nodeMap
         */
        function initAllNodes(node, nodeMap) {

            allNodes.add(node);

            // Recurs. call for node's children.
            let childrenPaths = core.getChildrenPaths(node);

            logger.info('paths', childrenPaths);

            for (let j = 0; j < childrenPaths.length; j += 1) {
                let childNode = nodeMap[childrenPaths[j]];

                initAllNodes(childNode, nodeMap);
            }
        }

        /**
         * Return the json data for a component
         * @param node
         * @param nodeMap
         * @returns {{name: string, isMeta: string, metaType: string}}
         */
        function getComponentJsonData(node, nodeMap){
            // set up part of JSON
            let nodeData = {
                name: '',
                isMeta: '',
                metaType: ''
            };

            nodeData.children = {};

            // get all of the valid attributes for the node
            let attrs = core.getValidAttributeNames(node);
            for (let i = 0; i < attrs.length; i += 1) {
                nodeData[attrs[i]] = core.getAttribute(node, attrs[i]);
            }

            // get node's metatype
            let metaTypeOfNode = getMetaType(node);
            nodeData.metaType = core.getAttribute(metaTypeOfNode, 'name');

            nodeData.guid = core.getGuid(node);

            return nodeData
        }

        // Return the data for a node (except ROOT node)
        function getNodeData(node, nodeMap) {

            if(core.getAttribute(node, 'name') === "TrainingData"){

                jsonTree.madeit = "yes";

                let floc = core.getAttribute(node, 'FileLocation');

                let dataTemplate = String(core.getAttribute(node, 'EJSTemplate'));
                // let dataTemplate = "<%= outputVar1 %> = <%=sparkVar%>.read.load('<%=FileLocation%>')"
                let dataNode = {
                    inputFile: floc,
                    guid: "123456data",
                    requirements: []
                };

                let sparkVar = 'spark';

                jsonTree.templ = dataTemplate;
                jsonTree.check2 = ejs.render(dataTemplate, {DataOut: "output", sparkVar: "spark", FileLocation: "file"});

            }

            // set up part of JSON
            let nodeData = {
                name: '',
                isMeta: '',
                metaType: ''
            };

            nodeData.children = {};

            // get all of the valid attributes for the node
            let attrs = core.getValidAttributeNames(node);
            for (let i = 0; i < attrs.length; i += 1) {
                nodeData[attrs[i]] = core.getAttribute(node, attrs[i]);

            }

            // get node's metatype
            let metaTypeOfNode = getMetaType(node);
            nodeData.metaType = core.getAttribute(metaTypeOfNode, 'name')

            // Recurs. call for node's children.
            let childrenPaths = core.getChildrenPaths(node);

            for (let j = 0; j < childrenPaths.length; j += 1) {

                let childNode = nodeMap[childrenPaths[j]];
                let childNodeData = getNodeData(childNode, nodeMap);
                let childRelId = core.getRelid(childNode);

                nodeData.children[childRelId] = childNodeData;
            }

            // If meta node:
            if (isMeta(node)) {
                addMetaNode(node)
                nodeData.isMeta = 'true'
            } else {
                nodeData.isMeta = 'false'
            }

            // check to see if it's a connection
            if (self.isMetaTypeOf(node, self.META.Connection) && !isMeta(node)) {

                let srcPath = core.getPointerPath(node, 'src');
                let dstPath = core.getPointerPath(node, 'dst');

                if (srcPath && dstPath) {

                    let srcNode = nodeMap[srcPath];
                    let dstNode = nodeMap[dstPath];

                    let srcParent = core.getParent(srcNode);
                    let dstParent = core.getParent(dstNode);

                    nodeData.src = core.getAttribute(srcParent, 'name') + '.' + core.getAttribute(srcNode, 'name');
                    nodeData.dst = core.getAttribute(dstParent, 'name') + '.' + core.getAttribute(dstNode, 'name');
                }
            }

            // logger.info(JSON.stringify(nodeData))

            return nodeData
        }

        // Main execution sequence
        this
            .loadNodeMap(activeNode)
            .then((nodeMap) => {

                let rootChildren = core.getChildrenPaths(activeNode);

                // Init the allNodes variable
                for (let i = 0; i < rootChildren.length; i += 1) {
                    let childNode = nodeMap[rootChildren[i]];
                    initAllNodes(childNode, nodeMap)
                }

                // Get set of all node components and inititilize the componentDataDictionary:
                let allComponents = getAllComponents(allNodes, nodeMap);

                // Get the sequence of nodes to convert to code
                let sequence = getSequenceGraph(allComponents, componentDataDictionary);

                // Convert the sequence of nodes into spark code
                let output = createSparkCodeFromSequence(sequence, componentDataDictionary);

                logger.info(output);

                artifact = self.blobClient.createArtifact('results');
                return artifact.addFiles({
                    'testPython.py': output
                });

            }).then((metadataHash) => {

            logger.info(metadataHash);
            self.result.setSuccess(true);
            return artifact.save();

        }).then(artifactHash => {

            self.result.addArtifact(artifactHash);
            self.result.setSuccess(true);
            callback(null, self.result);

        }).catch(function (err) {

            logger.error(err);
            callback(err);
        });
        //})};
    };


    return SparkCodeGen;
});