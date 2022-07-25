#include "huffman.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;


/**
 * TODO Complete this function
 **/
int huffman_encode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	int freqArray[256] = {};
	determineFrequencies(bufin,bufinlen,freqArray);
	
	//determine how many code symbols
	int numLeafNodes = 0;
	for (int i = 0; i < 256; i++)
	{
		if (freqArray[i] != 0)
		{
			numLeafNodes++;
		}
	}
	huffmanTree tree;
	//allocate space for nodes
	//number of internal nodes is one less than the number of leaf nodes since one is eliminated each time
	huffmanNode nodes[2*numLeafNodes-1] = {};
	tree.nodes = nodes;
	createHuffmanTree(freqArray,&tree,numLeafNodes);

	vector<char> codeTable[256] = {};

	createCodeTable(&tree,codeTable);

	//print code table
	cout << "Code Table:\n";
	for (int i = 0; i < 256; i++)
	{
		cout << i << ": ";
		for (int j = 0; j < codeTable[i].size(); j++)
		{
			cout << codeTable[i][j];
		}
		cout << "\n";
	}
	

	//since we know the frequencies of each character,
	//using the code lengths we can determine the required length of the output buffer.
	unsigned int outlen = 0;
	for (int i = 0; i < 256; i++)
	{
		outlen += freqArray[i]*codeTable[i].size();
	}
	//convert bits to bytes
	outlen = (int) ceil(outlen/8.0);
	//add required size for frequency table (256*4 = 1024)
	outlen += 1024;

	unsigned char *outputBuffer = (unsigned char*) malloc(outlen*sizeof(unsigned char));
	//write frequency table
	int outIdx = 0;
	for (int i = 0; i < 256; i ++)
	{
		outputBuffer[outIdx++] = (unsigned char)((freqArray[i] >> 24) & 0xFF);
		outputBuffer[outIdx++] = (unsigned char)((freqArray[i] >> 16) & 0xFF);
		outputBuffer[outIdx++] = (unsigned char)((freqArray[i] >> 8) & 0xFF);
		outputBuffer[outIdx++] = (unsigned char)(freqArray[i] & 0xFF);
	}

	//outputBuffer[outIdx] = leftoverBits;

	createOutputBytes(bufin,bufinlen,outputBuffer,outlen,codeTable);

	*pbufoutlen = outlen;
	*pbufout = outputBuffer;

	cout << "Encoding complete\n";
	return 0;
}


/**
 * TODO Complete this function
 **/
int huffman_decode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	int freqArray[256] = {};
	int outlen = 0;
	//read frequency array
	for (int i = 0; i < 256; i++)
	{
		freqArray[i] = (bufin[4*i] << 24) | (bufin[4*i+1] << 16) + (bufin[4*i+2] << 8) + bufin[4*i+3];
		outlen += freqArray[i];
	}

	//determine how many code symbols
	int numLeafNodes = 0;
	for (int i = 0; i < 256; i++)
	{
		if (freqArray[i] != 0)
		{
			numLeafNodes++;
		}
	}
	huffmanTree tree;
	//allocate space for nodes
	//number of internal nodes is one less than the number of leaf nodes since one is eliminated each time
	huffmanNode nodes[2*numLeafNodes-1] = {};
	tree.nodes = nodes;
	createHuffmanTree(freqArray,&tree,numLeafNodes);

	unsigned char *outputBuffer = (unsigned char*) malloc((outlen+1)*sizeof(unsigned char));

	int outIdx = 0;
	int byteNum = 1024;
	int bitNum = 0;
	huffmanNode *root = tree.root;
	huffmanNode *currNode = root;
	while ((byteNum < bufinlen) && (outIdx < outlen))
	{	
		unsigned char nextBit = (bufin[byteNum] >> (7 - bitNum)) & 0x01;
		if (nextBit == 0x00)
		{
			currNode = currNode->left;
		}
		else {
			currNode = currNode->right;
		}
		if (isLeaf(currNode))
		{
			outputBuffer[outIdx] = currNode->val;
			outIdx++;
			currNode = root;
		}
		bitNum++;
		if (bitNum >= 8)
		{
			bitNum = 0;
			byteNum++;
		}
	}
	outputBuffer[outIdx] = '\0'; //add null terminator

	*pbufoutlen = outlen+1;
	*pbufout = outputBuffer;

	cout << "Decoding complete\n";
	return 0;
}

void determineFrequencies(const unsigned char* bufin, unsigned int bufinlen, int *freqArray)
{
	for(int i = 0; i < bufinlen; i++)
	{
		freqArray[bufin[i]]++;
	}
	
	//print to test
	cout << "Frequency Array:\n";
	for(int j = 0; j < 256; j++)
	{
		cout << j << ": " << freqArray[j] << endl;
	}
	
	

}

void createHuffmanTree(int *freqArray, huffmanTree* tree, int numLeafNodes)
{
	huffmanNode *nodes = tree->nodes;

	int j = 0;
	int maxFreq = 0;

	for (int i = 0; i < 256; i++)
	{
		if (freqArray[i] != 0)
		{
			if (freqArray[i] > maxFreq)
			{
				maxFreq = freqArray[i];
			}
			huffmanNode *node = &nodes[j];
			node->val = i;
			node->freq = freqArray[i];
			node->left = nullptr;
			node->right = nullptr;
			j++;
		}
	}

	//radix sort
	radixSort(nodes,numLeafNodes,maxFreq,10);
	
	huffmanHeap minHeap;
	huffmanNode* heapArray[numLeafNodes] = {};
	minHeap.heap = &heapArray[0];
	minHeap.numNodes = 0;
	//start min heap with sorted nodes
	for (int k = 0; k < numLeafNodes; k++)
	{
		minHeap.heap[k] = &nodes[k];
	}
	minHeap.numNodes = numLeafNodes;

	//create intermediate nodes, starting with smallest frequencies
	j = numLeafNodes;

	while (minHeap.numNodes > 1)
	{
		//extract two lowest frequency nodes from the priority queue
		huffmanNode *minNode1;
		huffmanNode *minNode2; 
		getNextMinNode(&minHeap,&minNode1);
		getNextMinNode(&minHeap,&minNode2);

		//link new node to them
		huffmanNode *newNode = &nodes[j];
		newNode->val = 255;
		newNode->freq = minNode1->freq + minNode2->freq;
		newNode->left = minNode1;
		newNode->right = minNode2;

		//update root of tree
		tree->root = newNode;
		
		//add new node to heap
		addToMinHeap(&minHeap,newNode);
		j++;
	}
}

void radixSort(huffmanNode *nodes, int numNodes, int maxFreq, int base)
{
	for (int i = 1; maxFreq/i > 0; i *= base)
	{
		//create count array
		int counts[base] = {};
		for(int j = 0; j < numNodes; j++)
		{
			counts[((nodes[j].freq)/i) % base]++;
		}

		//update to cumulative counts
		for(int j = 1; j < base; j++)
		{
			counts[j] += counts[j-1];
		}

		//rearrange nodes based on count array
		huffmanNode temp[numNodes] = {};
		int countsIndex;
		for(int j = numNodes-1; j >= 0; j--)
		{
			countsIndex = ((nodes[j].freq)/i) % base;
			temp[counts[countsIndex] - 1] = nodes[j];
			counts[countsIndex]--;
		}
		//replace original array
		for(int j = 0; j < numNodes; j++)
		{
			nodes[j] = temp[j];
		}
	}

	//print to test
	/*
	cout << "after" << endl;
	for(int j = 0; j < numNodes; j++)
	{
		cout << j << ": " << nodes[j].val << " " << nodes[j].freq << endl;
	}
	*/
}

void getNextMinNode(huffmanHeap *minHeap, huffmanNode **minNode)
{
	if (minHeap->numNodes == 0)
	{
		return;
	}
	*minNode = minHeap->heap[0];
	minHeap->heap[0] = minHeap->heap[(minHeap->numNodes)-1];
	minHeap->numNodes--;
	reHeap(minHeap,0);

}

void reHeap(huffmanHeap *minHeap, int index)
{
	//check if there are child nodes
	if (2*index+1 >= minHeap->numNodes)
	{
		return;
	}
	else if (2*index+2 >= minHeap->numNodes)
	{
		if (minHeap->heap[index]->freq>minHeap->heap[2*index+1]->freq)
		{
			swapNodes(minHeap,index,2*index+1);
		}
		return; //only one child node here guarantees no further children
	}

	//recursive cases
	int leftIndex = 2*index+1;
	int rightIndex = leftIndex+1;
	unsigned int currVal = minHeap->heap[index]->freq;
	unsigned int leftVal = minHeap->heap[leftIndex]->freq;
	unsigned int rightVal = minHeap->heap[rightIndex]->freq;

	if ((currVal < leftVal) && (currVal < rightVal))
	{
		return; //min is in appropriate place
	}
	else if (leftVal < rightVal)
	{
		swapNodes(minHeap,index,leftIndex);
		reHeap(minHeap,leftIndex);
	}
	else if (leftVal >= rightVal)
	{
		swapNodes(minHeap,index,rightIndex);
		reHeap(minHeap,rightIndex);
	}
	else
	{
		//should never be here
		cout << "ERROR in reHeap\n";
	}
}

void swapNodes(huffmanHeap *minHeap, int index1, int index2)
{
	huffmanNode *temp;
	temp = minHeap->heap[index1];
	minHeap->heap[index1] = minHeap->heap[index2];
	minHeap->heap[index2] = temp;
}

void addToMinHeap(huffmanHeap *minHeap, huffmanNode *newNode)
{
	int index = minHeap->numNodes;
	unsigned int currVal = newNode->freq;
	minHeap->heap[index] = newNode;
	minHeap->numNodes++;
	
	//swap nodes to maintain minHeap property
	int parIndex = (index-1)/2;
	unsigned int parVal = minHeap->heap[parIndex]->freq;
	while (currVal < parVal)
	{
		swapNodes(minHeap,index,parIndex);
		index = parIndex;
		if (index <= 0)
		{
			break;
		}
		parIndex = (index-1)/2;
		parVal = minHeap->heap[parIndex]->freq;
	}

}

void createCodeTable(huffmanTree *tree, vector<char> *codeTable)
{
	//traverse tree and assign codes, starting at root
	huffmanNode *n = tree->root;
	vector<char> v;
	traverseTree(n,codeTable,v);
	
}

bool isLeaf(huffmanNode *n)
{
	return !((n->left) || n->right);
}

void traverseTree(huffmanNode *n, vector<char> *codeTable, vector<char> v)
{
	if(isLeaf(n))
	{
		//assign a code to leaf nodes
		for(int i = 0; i < v.size(); i++)
		{
			codeTable[n->val].push_back(v.at(i));
		}
		return;
	}
	//left = 0, right = 1
	if(n->left)
	{
		v.push_back('0');
		traverseTree(n->left,codeTable,v);
		v.pop_back();
	}
	if(n->right)
	{
		v.push_back('1');
		traverseTree(n->right,codeTable,v);
		v.pop_back();
	}

	
}

void createOutputBytes(const unsigned char *bufin, int bufinlen, unsigned char* outputBuffer, int outlen, vector<char> *codeTable)
{
	int outIdx = 1024; //start after the overhead bytes

	//convert to bits, once a byte is complete, write to the buffer
	unsigned char nextByte = 0x00;
	int numBits = 0;

	for (int i = 0; i < bufinlen; i++)
	{
		vector<char> v = codeTable[bufin[i]];
		for (int j = 0; j < v.size(); j++)
		{
			unsigned char nextBit = (v.at(j) == '1') ? 0x01 : 0x00;
			nextByte = (nextByte << 1) | nextBit;
			numBits++;
			if (numBits == 8)
			{
				outputBuffer[outIdx] = nextByte;
				outIdx++;
				numBits = 0;
			}
		}
		
	}
	//fill leftover bits if needed
	if (numBits > 0)
	{
		nextByte <<= (8-numBits);
		outputBuffer[outIdx] = nextByte;
	}

}
