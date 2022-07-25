
#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stdio.h>
#include <stdint.h>
#include <vector>

struct huffmanNode {
	unsigned char val;
	unsigned int freq;
	huffmanNode* left;
	huffmanNode* right;
} ;

struct huffmanHeap {
	unsigned int numNodes;
	huffmanNode** heap;
} ;

struct huffmanTree{
	huffmanNode* root;
	huffmanNode* nodes;
} ;


/**
 * @param bufin       Array of characters to encode
 * @param bufinlen    Number of characters in the array
 * @param pbufout     Pointer to unallocated output array
 * @param pbufoutlen  Pointer to variable where to store output size
 *
 * @return error code (0 is no error)
 **/
int huffman_encode(const unsigned char *bufin,
		   uint32_t bufinlen,
		   unsigned char **pbufout,
		   uint32_t *pbufoutlen);


/**
 * @param bufin       Array of characters to decode
 * @param bufinlen    Number of characters in the array
 * @param pbufout     Pointer to unallocated output array
 * @param pbufoutlen  Pointer to variable where to store output size
 *
 * @return error code (0 is no error)
 **/
int huffman_decode(const unsigned char *bufin,
  		   uint32_t bufinlen,
		   unsigned char **bufout,
		   uint32_t *pbufoutlen);

uint32_t test(uint32_t testint);
void determineFrequencies(const unsigned char* bufin, unsigned int bufinlen, int *freqArray);
void createHuffmanTree(int *freqArray, huffmanTree *tree);
void radixSort(huffmanNode *nodes, int numNodes, int maxFreq, int base);
void getNextMinNode(huffmanHeap *minHeap, huffmanNode **minNode);
void reHeap(huffmanHeap *minHeap, int index);
void swapNodes(huffmanHeap *minHeap, int index1, int index2);
void addToMinHeap(huffmanHeap *minHeap, huffmanNode *newNode);
void createCodeTable(huffmanTree *tree, std::vector<char> *codeTable);
bool isLeaf(huffmanNode *n);
void traverseTree(huffmanNode *n, std::vector<char> *codeTable, std::vector<char> v);
void createHuffmanTree(int *freqArray, huffmanTree* tree, int numLeafNodes);
void createOutputBytes(const unsigned char *bufin, int bufinlen, unsigned char *outputBuffer, int outlen, std::vector<char> *codeTable);

#endif
